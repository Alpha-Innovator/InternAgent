import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Any, Dict, List
import json

# 导入日志模块
from utils.logger import get_logger

from agents.base_agent import BaseAgent
from utils.graph import DirectedGraph, NodeExecutionStatus
from utils.prompt_loader import load_prompt
from tools import get_tool_manager

from models import get_model

from datetime import datetime

# SPM (Strategy-Procedural Memory, paper Sec. 2.4.1) — planner-side strategic priors.
# The same experience KB used by the executor also guides the planner toward globally
# coherent reasoning graphs. Imported lazily/safely so the planner still runs when the
# agent_kb service/package is unavailable.
try:
    from agent_kb.example import call_hybrid_search
except Exception:  # pragma: no cover - agent_kb is optional
    call_hybrid_search = None


def _is_truthy(val) -> bool:
    """True for boolean True or the strings 'true'/'True' (env-var friendly)."""
    return val is True or (isinstance(val, str) and val.strip().lower() == "true")


def _need_memory_enabled(config) -> bool:
    """Resolve the SPM on/off switch: env `need_memory` takes precedence, then config."""
    cfg = config or {}
    return _is_truthy(os.getenv("need_memory", cfg.get("need_memory", False)))


class GlobalPlannerAgent(BaseAgent):
    """
    全局规划器 Agent，用于将自然语言任务拆解为子任务并构建依赖关系图
    """
    
    def __init__(self, model: str = "deepseek-r1", tools: List[Dict[str, Any]] = None, config=None, **model_kwargs):
        """
        初始化全局规划器 Agent
        
        Args:
            model: 使用的模型名称
            tools: 可用的工具列表
            config: GlobalPlannerConfig配置对象
            **model_kwargs: 传递给模型的参数
        """
        super().__init__(model=model, tools=tools, config=config)
        self.model_instance = get_model(model, **model_kwargs)
        self.tool_manager = get_tool_manager(config=config)
        self.logger = get_logger("GlobalPlannerAgent")
        self._set_logger(self.logger)  # 设置logger到基类
        self.graph = None
        self.config = config or {}
        
        # 从配置加载prompt（如果配置中没有指定，使用默认值）
        self.prompt_template = load_prompt(
            self.config,
            default_name="GLOBAL_PLANNER_PROMPT"
        )

        # SPM planner-side: strategic priors block, appended to the base prompt only when
        # SPM is active. Kept separate so the default planner prompt is unchanged when off.
        planner_memory_block_config = {
            "prompt_path": self.config.get("planner_memory_block_path"),
            "prompt_name": self.config.get("planner_memory_block_name")
        }
        self.planner_memory_block = load_prompt(
            planner_memory_block_config,
            default_name="GLOBAL_PLANNER_MEMORY_BLOCK"
        )
        self.prompt_template_with_memory = self.prompt_template + "\n" + self.planner_memory_block
        self.need_memory = _need_memory_enabled(self.config)
        self.memory_prompt = None  # external override hook (set_memory_prompt)
        self.base_retrieve_score_threshold = self.config.get('base_retrieve_score_threshold', 0.3)
        self.append_retrieve_score_threshold = self.config.get('append_retrieve_score_threshold', 0.65)

        self.max_iter = self.config.get('max_iter', 5)
        self.max_retries = self.config.get('max_retries', 3)
        self.max_nodes = self.config.get('max_nodes', 8)

    def set_memory_prompt(self, memory_prompt: str):
        """外部注入 SPM 记忆提示（覆盖内部检索）。paper Sec. 2.4.1 的 planner-side 钩子。"""
        self.memory_prompt = memory_prompt

    def _retrieve_planner_memory(self, question: str):
        """
        从 agent_kb 检索与当前问题相似的历史 planning 经验（strategic priors），
        组装成注入 planner prompt 的 memory_prompt。检索不可用或失败时返回 None（不影响规划）。
        """
        if not (self.need_memory and call_hybrid_search is not None):
            if self.need_memory and call_hybrid_search is None:
                self.logger.warning("need_memory 已开启，但 agent_kb 客户端不可用，planner 不使用记忆")
            return None
        try:
            base_similar = call_hybrid_search('base', question, top_k=2, weights={"text": 0.3, "semantic": 0.7})
            append_similar = call_hybrid_search('append', question, top_k=2, weights={"text": 0.3, "semantic": 0.7})
            refs = []
            for exp in (base_similar or []):
                if exp.get('total_score', 0) > self.base_retrieve_score_threshold:
                    refs.append((exp.get('query'), exp.get('agent_experience', '') or ''))
            for exp in (append_similar or []):
                if exp.get('total_score', 0) > self.append_retrieve_score_threshold:
                    refs.append((exp.get('query'), exp.get('agent_experience', '') or ''))
            if not refs:
                return None
            parts = []
            for q, planning_exp in refs:
                parts.append(f"[Similar past task]\n{q}\n[Planning experience]\n{planning_exp}")
            self.logger.info(f"SPM planner 检索到 {len(refs)} 条 planning 经验")
            return "\n\n".join(parts)
        except Exception as e:
            self.logger.warning(f"SPM planner 检索失败，本次规划不使用记忆: {e}")
            return None
    
    def build_graph_from_plan(self, plan_result: Dict[str, Any]) -> DirectedGraph:
        """
        从规划结果构建有向图
        
        Args:
            plan_result: 包含 nodes 和 edges 的规划结果
            
        Returns:
            DirectedGraph: 构建好的有向图
        """
        graph = DirectedGraph()
        
        # 添加节点
        nodes = plan_result.get('nodes', [])
        for node in nodes:
            node_id = node['node_id']
            task = node['task']
            type = node['type']
            graph.add_node(node_id, task=task, type=type)
        
        # 添加边
        edges = plan_result.get('edges', [])
        for edge in edges:
            source, target, relationship = edge['from'], edge['to'], edge['relationship']
            graph.add_edge(source, target, relationship=relationship)

        if graph.has_cycle():
            return None
        
        self.graph = graph
        return graph
    
    def get_execution_order(self) -> List[str]:
        """
        获取任务的执行顺序（拓扑排序）
        
        Returns:
            List[str]: 按执行顺序排列的节点ID列表
        """
        if self.graph is None:
            return []
        
        return self.graph.get_topological_sort() or []
    
    def get_ready_nodes(self) -> List[str]:
        """
        获取当前可执行的节点（没有前置依赖的节点）
        
        Returns:
            List[str]: 可执行节点ID列表
        """
        if self.graph is None:
            return []
        
        return self.graph.get_ready_nodes()
    
    def mark_node_executed(self, node_id: str) -> bool:
        """
        标记节点为已执行
        
        Args:
            node_id: 节点ID
            
        Returns:
            bool: 是否标记成功
        """
        if self.graph is None:
            return False
        
        return self.graph.set_node_status(node_id, NodeExecutionStatus.EXECUTED)
    
    def execute_one_step(self, input_data: str, question: str, current_iter = 1, max_iter = 5, additional_info = None, tools = None, memory_prompt = None) -> Dict[str, Any]:
        """
        执行任务规划，将自然语言任务拆解为子任务并构建依赖关系图

        Args:
            input_data: 中间图，包含nodes和edges
            question: 问题描述
            current_iter: 当前迭代次数
            max_iter: 最大迭代次数
            memory_prompt: SPM 检索到的 planning 先验（可选）；提供时使用带记忆的 prompt

        Returns:
            包含 nodes 和 edges 的字典，表示任务分解结果
        """

        if memory_prompt:
            formatted_prompt = self.prompt_template_with_memory.format(
                graph=input_data,
                question=question,
                max_iter=str(max_iter),
                current_iter=str(current_iter),
                additional_info=additional_info,
                tools=json.dumps(tools),
                max_nodes=str(self.max_nodes),
                memory_prompt=memory_prompt
            )
        else:
            formatted_prompt = self.prompt_template.format(
                graph=input_data,
                question=question,
                max_iter=str(max_iter),
                current_iter=str(current_iter),
                additional_info=additional_info,
                tools=json.dumps(tools),
                max_nodes=str(self.max_nodes)
            )

        # 调用模型生成响应
        response = self._generate_plan(formatted_prompt)

        return response

    def execute(self, input_data: str, file_path = None, additional_info = None) -> Dict[str, Any]:
        """
        执行任务规划，将自然语言任务拆解为子任务并构建依赖关系图
        
        Args:
            input_data: 自然语言描述的任务
            file_path: 文件路径
            additional_info: 额外的文档信息（可选）
            
        Returns:
            包含 nodes 和 edges 的字典，表示任务分解结果
        """

        max_iter = self.max_iter

        graph = {
            "nodes":[
                {"node_id":"task", "type":"answer", "task": input_data}
            ]
        }

        if additional_info:
            self.logger.info(f"接收到文档信息，长度: {len(additional_info)}")

        tools = self.tool_manager.list_tools()
        tool_info_list = []
        for tool in tools:
            tool_info = self.tool_manager.get_simple_tool_info(tool)
            tool_info_list.append(tool_info)

        # SPM planner-side: 取一次 planning 先验（外部注入优先，否则从 agent_kb 检索），
        # 整个规划循环复用，避免每轮重复检索。
        memory_prompt = self.memory_prompt or self._retrieve_planner_memory(input_data)

        while True:  # 检查生成的图中是否存在环，如果存在环，则重新构建图
            idx = 1
            while idx <= max_iter:
                # 添加重试机制
                max_retries = self.max_retries
                retry_count = 0
                response = None

                while retry_count < max_retries:
                    try:
                        response = self.execute_one_step(json.dumps(graph), input_data, current_iter=idx, max_iter=max_iter, additional_info=additional_info, tools=tool_info_list, memory_prompt=memory_prompt)
                        if response is not None:
                            break  # 成功获得响应，跳出重试循环
                    except Exception as e:
                        self.logger.warning(f"Global Planner Attempt {retry_count + 1} failed: {e}")
                    
                    retry_count += 1
                    if retry_count < max_retries:
                        self.logger.info(f"Global Planner Retrying... (attempt {retry_count + 1}/{max_retries})")
                
                if response is None:
                    self.logger.error(f"Failed to get response after {max_retries} attempts in iteration {idx}")
                    break
                
                self.logger.info(f"global planner graph in {idx} iteration: {response}")
                if response == "end":
                    self.logger.info("globa planner graph构建完成")
                    break
                elif response == graph:
                    self.logger.info("global planner graph构建完成")
                    break
                graph = response
                
                idx += 1
            
            # 构建图结构
            if response:
                graph = self.build_graph_from_plan(graph)
                if graph is None:
                    self.logger.warning("global planner graph存在环，重新构建")
                    continue
                else:
                    break

        return response
    
    def _generate_plan(self, prompt: str) -> Dict[str, Any]:
        """
        生成任务规划
        
        Args:
            prompt: 格式化的提示词
            
        Returns:
            任务规划结果（包含 nodes 和 edges）
        """
        try:
            # 调用模型生成规划
            response = self.model_instance.generate(prompt)  

            try:
                response = json.loads(response)
                # self.logger.info(f"global planner response: {response}")
                if response.get("need_modify") == True:
                    return response.get("graph")
                else:
                    return "end"
            except Exception as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                return None
            
        except Exception as e:
            self.logger.error(f"Model call failed: {e}")
            return None
    
    def plan_task(self, task_description: str) -> Dict[str, Any]:
        """
        规划任务的便捷方法
        
        Args:
            task_description: 任务描述
            
        Returns:
            任务规划结果
        """
        return self.execute(task_description)
    
    def get_prompt_template(self) -> str:
        """
        获取当前使用的提示词模板
        
        Returns:
            提示词模板
        """
        return self.prompt_template


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    agent = GlobalPlannerAgent()
    response = agent.execute("我想了解关于AI Agent的最新研究进展")
    agent.logger.info("任务规划结果:")
    agent.logger.info(json.dumps(response, ensure_ascii=False, indent=2))
    
    # 显示执行顺序
    execution_order = agent.get_execution_order()
    agent.logger.info(f"\n执行顺序: {execution_order}")
    
    # 显示当前可执行的节点
    ready_nodes = agent.get_ready_nodes()
    agent.logger.info(f"当前可执行节点: {ready_nodes}")
    
    # 文本形式可视化（确保中文能正确显示）
    agent.logger.info("\n正在生成文本形式可视化...")
    agent.graph.visualize_text()
    
    # 图形形式可视化
    agent.logger.info("\n正在生成图形可视化...")
    try:
        agent.graph.visualize_graph(save_path=f"task_dependency_graph_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    except Exception as e:
        agent.logger.error(f"图形可视化失败: {e}")
        agent.logger.info("已使用文本形式可视化作为备用方案")