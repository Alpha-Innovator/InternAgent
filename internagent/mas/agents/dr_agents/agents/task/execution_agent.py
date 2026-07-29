import sys
import os
import json
import re
from typing import Any, Dict, List, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.base_agent import BaseAgent
from models import get_model
from tools import get_tool_manager
from utils.reference_manager import ReferenceManager
from utils.logger import get_logger
from utils.prompt_loader import load_prompt

# Strategy-Procedural Memory (SPM, paper Sec. 2.4.1) — optional experience KB.
# Imported lazily/safely so the DR path still runs when the agent_kb service or
# package is unavailable. When missing, SPM retrieval is skipped and the normal
# EXECUTION_PROMPT is used.
try:
    from agent_kb.example import call_hybrid_search
except Exception:  # pragma: no cover - agent_kb is optional
    call_hybrid_search = None


def _is_truthy(val) -> bool:
    """True for boolean True or the strings 'true'/'True' (env-var friendly)."""
    return val is True or (isinstance(val, str) and val.strip().lower() == "true")


def _need_memory_enabled(config: Dict[str, Any]) -> bool:
    """Resolve the SPM on/off switch: env `need_memory` takes precedence, then config."""
    return _is_truthy(os.getenv("need_memory", config.get("need_memory", False)))


class ExecutionAgent(BaseAgent):
    """
    执行代理，负责调用工具并处理工具调用结果
    """
    
    def __init__(self, model: str = "deepseek-r1", tools: List[Dict[str, Any]] = None, tool_mode: str = "local", config=None, summarizer_model: str = None, reference_manager=None, **model_kwargs):
        """
        初始化执行代理
        
        Args:
            model: 使用的模型名称
            tools: 可用的工具列表
            tool_mode: 工具模式，'test'使用测试工具，其他值使用本地工具
            config: TaskWorkflowConfig配置对象
            summarizer_model: 用于总结的模型名称（如果为None则使用model）
            reference_manager: 参考文献管理器（如果为None则创建新的）
            **model_kwargs: 传递给模型的参数
        """
        super().__init__(model=model, tools=tools)
        self.model_instance = get_model(model, **model_kwargs)
        self.tool_mode = tool_mode
        # 使用基类的 _logger，可以通过 _set_logger 动态更改
        self._set_logger(get_logger("ExecutionAgent"))
        self.tool_manager = get_tool_manager(mode=tool_mode, config=config)
        self.tool_map = self._build_tool_map()
        self.model_kwargs = model_kwargs
        self.config = config or {}
        
        # 从配置加载execution prompt（如果配置中没有指定，使用默认值）
        execution_config = {
            "prompt_path": self.config.get("execution_prompt_path"),
            "prompt_name": self.config.get("execution_prompt_name")
        }
        self.execution_prompt = load_prompt(
            execution_config,
            default_name="EXECUTION_PROMPT"
        )
        
        # 从配置加载summary prompt（如果配置中没有指定，使用默认值）
        summary_config = {
            "prompt_path": self.config.get("summary_prompt_path"),
            "prompt_name": self.config.get("summary_prompt_name")
        }
        self.summary_prompt = load_prompt(
            summary_config,
            default_name="SUMMARY_PROMPT"
        )

        # SPM (Strategy-Procedural Memory): memory-augmented execution prompt +
        # retrieval switch. Only takes effect when need_memory is set AND the
        # agent_kb client is importable; otherwise the normal prompt is used.
        execution_memory_config = {
            "prompt_path": self.config.get("execution_prompt_with_memory_path"),
            "prompt_name": self.config.get("execution_prompt_with_memory_name")
        }
        self.execution_prompt_with_memory = load_prompt(
            execution_memory_config,
            default_name="EXECUTION_PROMPT_WITHMEMORY"
        )
        self.need_memory = _need_memory_enabled(self.config)
        self.base_retrieve_score_threshold = self.config.get('base_retrieve_score_threshold', 0.3)
        self.append_retrieve_score_threshold = self.config.get('append_retrieve_score_threshold', 0.65)

        self.max_tool_calls = self.config.get('max_tool_calls', 15)
        self.node_id = None  # 节点ID
        self.subtask_id = None  # 子任务ID
        # 使用传入的reference_manager或创建新的
        self.reference_manager = reference_manager if reference_manager is not None else ReferenceManager()
        self.summarizer_model = summarizer_model or model  # 如果没有指定summarizer_model，使用默认model
    
    def _build_tool_map(self) -> Dict[str, Dict[str, Any]]:
        """
        构建工具映射表
        
        Returns:
            工具名称到工具配置的映射
        """
        tool_map = {}
        
        # 如果提供了显式的工具列表，使用它们
        if self.tools:
            for tool in self.tools:
                if isinstance(tool, dict) and 'name' in tool:
                    tool_map[tool['name']] = tool
        else:
            # 否则从工具管理器获取工具
            self._logger.info(f"没有提供显式的工具列表，从工具管理器获取工具")
            available_tools = self.tool_manager.list_tools()
            for tool_name in available_tools:
                tool_info = self.tool_manager.get_tool_info(tool_name)
                if tool_info:
                    # 为工具创建标准格式的配置
                    tool_config = {
                        "name": tool_name,
                        "type": "function",
                        "function_name": tool_name,
                        "description": tool_info.get("doc", f"Tool: {tool_name}"),
                        "required_parameters": tool_info.get("required_parameters", []),
                        "parameters": tool_info.get("parameters", {})
                    }
                    tool_map[tool_name] = tool_config
        
        self._logger.info(f"构建工具映射，共 {len(tool_map)} 个工具: {list(tool_map.keys())}")
        return tool_map
    
    def execute_one_step(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行子任务，让模型选择并调用工具. 如果模型返回tool_calls，则调用工具，并返回工具调用结果, 然后结束, 如果需要反复执行, 直到模型返回tool_calls为空为止. 则需要调用execute方法.
        
        Args:
            input_data: 包含以下键的字典：
                - messages: 历史消息
                
        Returns:
            执行结果字典
        """
        messages = input_data.get("messages", [])
        
        # 从tool_map获取工具并转换为OpenAI格式
        openai_tools = self._convert_tool_map_to_openai_format()
        
        try:
            
            # 清理和验证消息格式
            cleaned_messages = []
            for msg in messages:
                cleaned_msg = self._validate_and_clean_message(msg)
                cleaned_messages.append(cleaned_msg)
                # self._logger.info(f"清理后消息: {cleaned_msg}")
            
            # 调用模型进行工具选择
            response = self.model_instance.generate_with_tools(
                messages=cleaned_messages,
                tools=openai_tools,
            )
            # self._logger.debug("模型返回结果: ", response)
            # 解析响应
            choice = response["choices"][0]
            message = choice["message"]
            
            # 检查是否有工具调用
            if "tool_calls" in message and message["tool_calls"]:
                self._logger.info(f"检测到 {len(message['tool_calls'])} 个工具调用")
                # 执行工具调用
                tool_results = []
                for i, tool_call in enumerate(message["tool_calls"]):
                    # 安全地获取工具名称
                    tool_name = self._get_tool_name_from_call(tool_call)
                    self._logger.info(f"执行工具调用 {i+1}/{len(message['tool_calls'])}: {tool_name}")
                    result = self._execute_tool_call_from_model(tool_call)
                    tool_results.append(result)
                
                # 不在这里修改messages，让调用方处理
                # 工具调用结果已经包含在tool_results中
                
                # 如果需要，可以继续对话
                return {
                    "tool_calls": tool_results,
                    "assistant_message": message  # 添加完整的助手消息
                }
            else:
                # 没有工具调用，直接返回文本响应
                self._logger.info("没有检测到工具调用，返回文本响应")
                return {
                    "tool_calls": [],
                    "assistant_message": message
                }
                
        except Exception as e:
            self._logger.error(f"Model execution failed: {e}")
            return {
                "error": str(e),
                "tool_calls": [],
                "summary": f"Execution failed: {str(e)}"
            }
    
    def execute(self, subtask: str, context: Dict[str, Any], use_planner: bool = True, query = None) -> Dict[str, Any]:
        """
        执行单个子任务，包含工具调用循环, 以及最后的总结
        
        Args:
            subtask: 子任务描述
            context: 上下文信息，包含：
                - task: 总体任务
                - history_subtasks: 历史子任务列表
                - knowledge_info: 知识信息
                - node_id: 节点ID（可选）
                - subtask_id: 子任务ID（可选）
                
        Returns:
            子任务执行结果
        """

        
        try:
            self._logger.info(f"开始执行子任务: {subtask}")

            # 预初始化，保证异常分支也能安全引用
            messages = []
            wrong_messages = []  # SPM: 记录失败的工具调用，供经验蒸馏使用

            # 从context中获取信息
            task = context.get("task", "")
            history_subtasks = context.get("history_subtasks", [])
            knowledge_info = context.get("knowledge_info", {})
            file_path = context.get("file_path", "")
            self.node_id = context.get("node_id", None)
            self.subtask_id = context.get("subtask_id", None)

            # SPM retrieval: when enabled and the KB client is available, retrieve
            # strategy/procedural experiences for similar tasks and inject them via
            # the memory-augmented prompt. Any failure degrades to the normal prompt.
            retrieved_pairs = []
            use_memory_prompt = False
            if self.need_memory and call_hybrid_search is not None:
                try:
                    self._logger.info("SPM 已启用，从 agent_kb 检索经验 =============== ")
                    base_similar = call_hybrid_search('base', task, top_k=2, weights={"text": 0.3, "semantic": 0.7})
                    append_similar = call_hybrid_search('append', task, top_k=2, weights={"text": 0.3, "semantic": 0.7})
                    for exp in (base_similar or []):
                        if exp.get('total_score', 0) > self.base_retrieve_score_threshold:
                            retrieved_pairs.append({
                                'retrieved_question': exp.get('query'),
                                'retrieved_experience': (exp.get('agent_experience', '') or '') + '\n\n' + (exp.get('search_agent_experience', '') or '')
                            })
                    for exp in (append_similar or []):
                        if exp.get('total_score', 0) > self.append_retrieve_score_threshold:
                            retrieved_pairs.append({
                                'retrieved_question': exp.get('query'),
                                'retrieved_experience': (exp.get('agent_experience', '') or '') + '\n\n' + (exp.get('search_agent_experience', '') or '')
                            })
                    use_memory_prompt = True
                except Exception as e:
                    self._logger.warning(f"SPM 检索失败，回退到普通 prompt: {e}")
                    use_memory_prompt = False
            elif self.need_memory and call_hybrid_search is None:
                self._logger.warning("need_memory 已开启，但 agent_kb 客户端不可用，使用普通 prompt")

            # 初始化消息历史，使用execution_prompt（或带记忆的变体）
            if use_memory_prompt:
                formatted_prompt = self.execution_prompt_with_memory.format(
                    task=task,
                    query=query,
                    history_subtasks=json.dumps(history_subtasks, ensure_ascii=False, indent=2) if history_subtasks else "[]",
                    subtask=subtask,
                    knowledge_info=json.dumps(knowledge_info, ensure_ascii=False, indent=2) if knowledge_info else "{}",
                    file_path=file_path,
                    retrieved_pairs=json.dumps(retrieved_pairs, ensure_ascii=False, indent=2) if retrieved_pairs else "[]",
                )
            else:
                formatted_prompt = self.execution_prompt.format(
                    task=task,
                    query=query,
                    history_subtasks=json.dumps(history_subtasks, ensure_ascii=False, indent=2) if history_subtasks else "[]",
                    subtask=subtask,
                    knowledge_info=json.dumps(knowledge_info, ensure_ascii=False, indent=2) if knowledge_info else "{}",
                    file_path=file_path
                )
            messages = [{"role": "user", "content": formatted_prompt}]
            tool_call_count = 0
            max_tool_calls = self.max_tool_calls  # 防止无限循环
            
            while tool_call_count < max_tool_calls:
                self._logger.info(f"执行工具调用循环 {tool_call_count + 1}/{max_tool_calls}")
                
                # 调用执行代理
                execution_input = {
                    "messages": messages
                }
                
                result = self.execute_one_step(execution_input)
                
                
                # 检查是否有工具调用
                if result.get("tool_calls") and len(result["tool_calls"]) > 0:
                    # self._logger.info(f"执行代理返回结果: {result.get('tool_calls', [])}")
                    
                    # 添加助手消息到历史（包含工具调用）
                    if "assistant_message" in result:
                        assistant_msg = result["assistant_message"]
                        # 将助手消息转换为可序列化的格式
                        serializable_assistant_msg = self._convert_assistant_message_to_serializable(assistant_msg)
                        self._logger.info(f"添加助手消息: {serializable_assistant_msg}")
                        messages.append(serializable_assistant_msg)
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": result.get("final_response", "")
                        }
                        messages.append(assistant_message)
                    
                    # 添加工具调用结果到消息历史
                    if "tool_calls" in result and result["tool_calls"]:
                        # 获取助手消息中的工具调用ID（以及名称/参数，供 SPM 记录失败调用）
                        assistant_tool_call_ids = []
                        assistant_tool_call_map = {}
                        if "assistant_message" in result and "tool_calls" in result["assistant_message"]:
                            for tool_call in result["assistant_message"]["tool_calls"]:
                                tc_id = self._get_tool_call_id(tool_call)
                                assistant_tool_call_ids.append(tc_id)
                                tc_name = self._get_tool_name_from_call(tool_call)
                                try:
                                    if hasattr(tool_call, 'function'):
                                        tc_args = getattr(tool_call.function, 'arguments', None)
                                    elif isinstance(tool_call, dict):
                                        tc_args = tool_call.get('function', {}).get('arguments')
                                    else:
                                        tc_args = None
                                except Exception:
                                    tc_args = None
                                assistant_tool_call_map[tc_id] = {
                                    "tool_name": tc_name,
                                    "tool_call_id": tc_id,
                                    "argument": tc_args,
                                }

                        # 使用工具调用结果中的ID
                        for i, tool_result in enumerate(result["tool_calls"]):
                            is_success = bool(tool_result.get("success", False))
                            content = str(tool_result.get("result", "")) if is_success else f"Error: {tool_result.get('error', 'Unknown error')}"

                            # 使用助手消息中的工具调用ID，如果没有则生成一个
                            if i < len(assistant_tool_call_ids):
                                tool_call_id = assistant_tool_call_ids[i]
                            else:
                                tool_name = tool_result.get("tool_name", f"tool_{i}")
                                tool_call_id = f"call_{tool_call_count}_{i}_{tool_name}"

                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": content
                            }
                            messages.append(tool_message)
                            self._logger.info(f"添加工具消息: {tool_message}")

                            # SPM: 记录失败的工具调用（含发起时的命令/参数），供经验蒸馏
                            if not is_success:
                                wrong_messages.append({
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_result.get("tool_name", ""),
                                    "command": assistant_tool_call_map.get(tool_call_id, {
                                        "tool_name": tool_result.get("tool_name", ""),
                                        "tool_call_id": tool_call_id,
                                        "raw_tool_call": None,
                                    }),
                                    "error": tool_result.get("error", "Unknown error"),
                                })
                    # 注意：这里不需要else分支，因为如果result["tool_calls"]为空，就不会进入上面的if分支
                    
                    tool_call_count += 1
                    
                    # 如果工具调用成功，继续循环
                    if any(tc.get("success", False) for tc in result["tool_calls"]):
                        # self._logger.info("工具调用成功，继续循环")    
                        continue
                    else:
                        # 所有工具调用都失败，停止循环
                        self._logger.warning("所有工具调用都失败，停止循环")
                        break
                        
                else:
                    # 没有工具调用，任务完成
                    # SPM: 保留最终 assistant 文本，使经验蒸馏能看到收尾推理
                    if "assistant_message" in result and result["assistant_message"].get("content"):
                        messages.append({
                            "role": "assistant",
                            "content": result["assistant_message"]["content"]
                        })
                    self._logger.info("没有检测到工具调用，子任务完成")
                    break

            # 生成最终响应
            final_result = self._generate_subtask_response(subtask, messages, context)
            try:
                # 尝试提取JSON（可能在文本中间或有额外内容）
                parsed_result = self._extract_and_parse_json(final_result)
                if parsed_result:
                    success = parsed_result.get("success", True)
                    summary = parsed_result.get("summary", "")
                else:
                    # 如果无法提取JSON，使用原始响应
                    summary = final_result
                    success = True
            except Exception as e:
                self._logger.warning(f"JSON解析失败，使用原始响应: {e}")
                # 如果JSON解析失败，使用原始响应作为summary
                summary = final_result
                success = True

            return {
                "subtask": subtask,
                "completed": True,
                "tool_calls": tool_call_count,
                "success": success,
                "messages": messages,
                "summary": summary,
                "wrong_messages": wrong_messages
            }

        except Exception as e:
            self._logger.error(f"子任务执行失败: {e}")
            return {
                "subtask": subtask,
                "completed": False,
                "error": str(e),
                "tool_calls": 0,
                "summary": f"执行失败: {str(e)}",
                "success": False,
                "messages": messages,
                "wrong_messages": wrong_messages
            }
    
    def _extract_and_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中提取并解析JSON，处理模型返回JSON后附加额外内容的情况
        
        Args:
            text: 可能包含JSON的文本
            
        Returns:
            解析后的字典，如果无法解析则返回None
        """
        try:
            # 首先尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试查找JSON对象的开始和结束
        try:
            # 查找第一个 { 和最后一个匹配的 }
            start_idx = text.find('{')
            if start_idx == -1:
                return None
            
            # 从第一个 { 开始，找到匹配的 }
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx == -1:
                return None
            
            # 提取并解析JSON
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
            
        except Exception as e:
            self._logger.debug(f"无法从文本中提取JSON: {e}")
            return None
    
    def _generate_subtask_response(self, subtask: str, messages: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """
        生成子任务的最终响应
        
        Args:
            subtask: 子任务描述
            messages: 消息历史
            context: 上下文信息
            
        Returns:
            最终响应
        """
        try:
            self._logger.info(f"开始生成子任务阶段总结: {subtask}")
            
            summary_prompt = self.summary_prompt.format(
                subtask=subtask,
                messages=json.dumps(messages, ensure_ascii=False, indent=2) if messages else "[]",
                knowledge_info=json.dumps(context.get("knowledge_info", {}), ensure_ascii=False, indent=2) if context.get("knowledge_info", {}) else "{}",
            ) 

            self._logger.info(f"子任务阶段总结提示: {summary_prompt}")
            
            # 调用summarizer模型生成总结
            self._logger.info(f"使用模型 {self.summarizer_model} 生成子任务总结")
            summarizer_instance = get_model(self.summarizer_model, **self.model_kwargs)
            final_response = summarizer_instance.generate(summary_prompt)
            
            self._logger.info(f"子任务阶段总结内容: {final_response}")
            
            return final_response
            
        except Exception as e:
            self._logger.error(f"生成子任务阶段总结失败: {e}")
            return f"子任务'{subtask}'执行完成，但生成总结时出现错误: {str(e)}"
    
    def _execute_single_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个工具调用
        
        Args:
            tool_call: 工具调用信息
            
        Returns:
            工具调用结果
        """
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        if not tool_name:
            raise ValueError("Tool name is required")
        
        # 查找工具
        tool_config = self.tool_map.get(tool_name)
        if not tool_config:
            raise ValueError(f"Tool '{tool_name}' not found in available tools")
        
        # 验证参数
        self._validate_tool_arguments(tool_config, arguments)
        
        # 执行工具
        try:
            result = self._call_tool(tool_config, arguments)

            self._logger.info(f"工具 {tool_name} 执行结果: {result}, 类型: {type(result)}")
            
            # 如果是搜索工具，提取URL并添加到参考文献管理器，同时获取修改后的result
            if self._is_search_tool(tool_name):
                extracted_urls, modified_result = self._extract_urls_from_result(result, tool_name)
                if extracted_urls:
                    self._logger.info(f"从工具 {tool_name} 中提取到 {len(extracted_urls)} 个URL")
                    # 使用修改后的result（包含tag信息）
                    result = modified_result
            
            # 发送工具调用成功事件
            if self.node_id is not None and self.subtask_id is not None:
                self.send_redis_event("get_subtask_process", {
                    "node_id": self.node_id,
                    "subtask_id": self.subtask_id,
                    "process": {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result,  # 限制结果长度，避免过大
                    }
                })
            
            return {
                "tool_name": tool_name,
                "success": True,
                "arguments": arguments,
                "result": result,
                "error": None
            }
        except Exception as e:
            return {
                "tool_name": tool_name,
                "success": False,
                "arguments": arguments,
                "result": None,
                "error": str(e)
            }
    
    def _validate_tool_arguments(self, tool_config: Dict[str, Any], arguments: Dict[str, Any]) -> None:
        """
        验证工具参数
        
        Args:
            tool_config: 工具配置
            arguments: 提供的参数
            
        Raises:
            ValueError: 参数验证失败
        """
        required_params = tool_config.get("required_parameters", [])
        for param in required_params:
            if param not in arguments:
                raise ValueError(f"Required parameter '{param}' is missing for tool '{tool_config['name']}'")
    
    def _call_tool(self, tool_config: Dict[str, Any], arguments: Dict[str, Any]) -> Any:
        """
        调用工具函数
        
        Args:
            tool_config: 工具配置
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        # 目前只支持函数类型工具
        tool_type = tool_config.get("type", "function")
        
        if tool_type == "function":
            return self._call_function_tool(tool_config, arguments)
        else:
            raise ValueError(f"Unsupported tool type: {tool_type}")
    
    def _call_function_tool(self, tool_config: Dict[str, Any], arguments: Dict[str, Any]) -> Any:
        """
        调用函数类型工具
        
        Args:
            tool_config: 工具配置
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        function_name = tool_config.get("name")
        
        try:
            # 使用工具管理器动态调用工具
            result = self.tool_manager.call_tool(function_name, **arguments)
            return result
        except Exception as e:
            self._logger.error(f"Tool execution failed for {function_name}: {e}")
            return f"Error executing {function_name}: {str(e)}"
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> str:
        """
        生成执行结果摘要
        
        Args:
            results: 执行结果列表
            
        Returns:
            摘要文本
        """
        successful_calls = sum(1 for result in results if result.get("success", False))
        total_calls = len(results)
        
        summary = f"Executed {total_calls} tool calls, {successful_calls} successful"
        
        if results:
            tool_names = [result.get("tool_name", "unknown") for result in results]
            summary += f". Tools used: {', '.join(tool_names)}"
        
        return summary
    
    def tool_call(self, tool_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行工具调用的便捷方法
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            context: 上下文信息
            
        Returns:
            工具调用结果
        """
        tool_call = {
            "name": tool_name,
            "arguments": arguments
        }
        
        return self._execute_single_tool_call(tool_call, context or {})
    
    def add_tool(self, tool: Dict[str, Any]) -> None:
        """
        添加工具并更新工具映射
        
        Args:
            tool: 工具配置字典
        """
        super().add_tool(tool)
        self.tool_map = self._build_tool_map()
    
    def get_available_tools(self) -> List[str]:
        """
        获取可用工具名称列表
        
        Returns:
            工具名称列表
        """
        return list(self.tool_map.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具信息
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具配置信息，如果不存在则返回None
        """
        return self.tool_map.get(tool_name)
    
    def _convert_tool_map_to_openai_format(self) -> List[Dict[str, Any]]:
        """
        将tool_map中的工具转换为OpenAI格式
        
        Returns:
            OpenAI格式的工具列表
        """
        openai_tools = []
        for tool_name, tool_config in self.tool_map.items():
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_config["name"],
                    "description": tool_config.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": tool_config.get("parameters", {}),
                        "required": tool_config.get("required_parameters", [])
                    }
                }
            }
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    def _convert_tools_to_openai_format(self) -> List[Dict[str, Any]]:
        """
        将工具转换为OpenAI格式（保留原方法以兼容）
        
        Returns:
            OpenAI格式的工具列表
        """
        openai_tools = []
        for tool in self.tools:
            if isinstance(tool, dict) and 'name' in tool:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": tool.get("required_parameters", [])
                        }
                    }
                }
                
                # 添加参数属性
                if "parameters" in tool:
                    openai_tool["function"]["parameters"]["properties"] = tool["parameters"]
                
                openai_tools.append(openai_tool)
        
        return openai_tools
    
    def _get_tool_name_from_call(self, tool_call: Any) -> str:
        """
        从工具调用对象中安全地获取工具名称
        
        Args:
            tool_call: 工具调用对象（可能是字典或ChatCompletionMessageToolCall对象）
            
        Returns:
            工具名称
        """
        try:
            if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                return tool_call.function.name
            elif isinstance(tool_call, dict) and 'function' in tool_call:
                return tool_call['function'].get('name', 'unknown')
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    
    def _get_tool_call_id(self, tool_call: Any) -> str:
        """
        从工具调用对象中安全地获取工具调用ID
        
        Args:
            tool_call: 工具调用对象（可能是字典或ChatCompletionMessageToolCall对象）
            
        Returns:
            工具调用ID
        """
        try:
            if hasattr(tool_call, 'id'):
                return tool_call.id
            elif isinstance(tool_call, dict) and 'id' in tool_call:
                return tool_call['id']
            else:
                return f"call_{id(tool_call)}"  # 生成一个唯一的ID
        except Exception:
            return f"call_{id(tool_call)}"  # 生成一个唯一的ID
    
    def _validate_and_clean_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证和清理消息格式，确保符合OpenAI API要求
        
        Args:
            message: 消息字典
            
        Returns:
            清理后的消息字典
        """
        cleaned_message = {}
        
        # 必需的字段
        if 'role' not in message:
            self._logger.warning("消息缺少role字段，设置为assistant")
            cleaned_message['role'] = 'assistant'
        else:
            cleaned_message['role'] = message['role']
        
        # content字段（可选，但如果有tool_calls则不应该有content）
        if 'content' in message and message['content'] is not None:
            # 确保content是字符串
            if not isinstance(message['content'], str):
                cleaned_message['content'] = str(message['content'])
            else:
                cleaned_message['content'] = message['content']
        elif 'tool_calls' not in message:
            # 如果没有tool_calls，确保有content
            cleaned_message['content'] = message.get('content', '')
        
        # tool_calls字段
        if 'tool_calls' in message and message['tool_calls']:
            cleaned_message['tool_calls'] = []
            for tool_call in message['tool_calls']:
                if isinstance(tool_call, dict):
                    cleaned_tool_call = {
                        'id': tool_call.get('id', f"call_{id(tool_call)}"),
                        'type': tool_call.get('type', 'function'),
                        'function': {
                            'name': tool_call.get('function', {}).get('name', 'unknown'),
                            'arguments': tool_call.get('function', {}).get('arguments', '{}')
                        }
                    }
                    cleaned_message['tool_calls'].append(cleaned_tool_call)
        
        # tool_call_id字段（用于tool角色消息）
        if 'tool_call_id' in message:
            cleaned_message['tool_call_id'] = str(message['tool_call_id'])
        
        return cleaned_message
    
    def _convert_assistant_message_to_serializable(self, assistant_msg: Any) -> Dict[str, Any]:
        """
        将助手消息转换为可序列化的格式
        
        Args:
            assistant_msg: 助手消息（可能包含ChatCompletionMessageToolCall对象）
            
        Returns:
            可序列化的助手消息字典
        """
        try:
            if isinstance(assistant_msg, dict):
                # 如果已经是字典，直接复制，但确保有role字段
                serializable_msg = assistant_msg.copy()
                if 'role' not in serializable_msg:
                    serializable_msg['role'] = 'assistant'
            else:
                # 如果是对象，转换为字典
                serializable_msg = {
                    "role": getattr(assistant_msg, 'role', 'assistant'),
                    "content": getattr(assistant_msg, 'content', '')
                }
            
            # 处理工具调用
            if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
                serializable_msg["tool_calls"] = []
                for tool_call in assistant_msg.tool_calls:
                    serializable_tool_call = {
                        "id": self._get_tool_call_id(tool_call),
                        "type": "function",
                        "function": {
                            "name": self._get_tool_name_from_call(tool_call),
                            "arguments": getattr(tool_call.function, 'arguments', '{}')
                        }
                    }
                    serializable_msg["tool_calls"].append(serializable_tool_call)
            elif isinstance(assistant_msg, dict) and "tool_calls" in assistant_msg:
                # 如果tool_calls已经是字典格式，确保它们是可序列化的
                serializable_msg["tool_calls"] = []
                for tool_call in assistant_msg["tool_calls"]:
                    if hasattr(tool_call, 'function'):
                        # 对象格式
                        serializable_tool_call = {
                            "id": self._get_tool_call_id(tool_call),
                            "type": "function",
                            "function": {
                                "name": self._get_tool_name_from_call(tool_call),
                                "arguments": getattr(tool_call.function, 'arguments', '{}')
                            }
                        }
                    else:
                        # 字典格式
                        serializable_tool_call = tool_call.copy()
                    serializable_msg["tool_calls"].append(serializable_tool_call)
            
            return serializable_msg
            
        except Exception as e:
            self._logger.error(f"转换助手消息失败: {e}")
            # 返回一个基本的可序列化消息
            return {
                "role": "assistant",
                "content": str(assistant_msg) if assistant_msg else ""
            }
    
    def _execute_tool_call_from_model(self, tool_call: Any) -> Dict[str, Any]:
        """
        执行来自模型的工具调用
        
        Args:
            tool_call: 模型返回的工具调用（可能是字典或ChatCompletionMessageToolCall对象）
            context: 上下文信息
            
        Returns:
            工具调用结果
        """
        try:
            # 处理不同类型的工具调用对象
            if hasattr(tool_call, 'function'):
                # ChatCompletionMessageToolCall对象
                function = tool_call.function
                tool_name = function.name
                arguments = function.arguments
            elif isinstance(tool_call, dict) and 'function' in tool_call:
                # 字典格式
                function = tool_call["function"]
                tool_name = function["name"]
                arguments = function["arguments"]
            else:
                return {
                    "tool_name": "unknown",
                    "success": False,
                    "error": f"Unsupported tool_call format: {type(tool_call)}",
                    "result": None
                }
            
            # 解析参数
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as e:
                    return {
                        "tool_name": tool_name,
                        "success": False,
                        "error": f"Invalid JSON arguments: {e}",
                        "result": None
                    }
            
            # 构建工具调用格式
            tool_call_data = {
                "name": tool_name,
                "arguments": arguments
            }
            
            return self._execute_single_tool_call(tool_call_data)
            
        except Exception as e:
            return {
                "tool_name": "unknown",
                "success": False,
                "error": f"Error processing tool_call: {str(e)}",
                "result": None
            }
    
    def _is_search_tool(self, tool_name: str) -> bool:
        """
        判断工具是否是搜索工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            是否是搜索工具
        """
        # 搜索工具列表：提取reference的工具
        allowed_tools = ['search_wiki', 'search_and_summarize_papers', 'search_and_summarize_webpages',
                         'search_tavily', 'search_volc', 'search_google',
                         'search_arxiv', 'search_openalex', 'search_crossref']
        return tool_name in allowed_tools
    
    
    def _extract_urls_from_result(self, result: Any, tool_name: str) -> tuple[List[str], Any]:
        """
        从工具结果中提取URL、标题和类型，并将tag添加到原始result中

        根据不同工具的返回格式处理：
        - search_wiki: 返回 dict，包含 url 和 title
        - search_and_summarize_papers: 返回 list，每个元素是 dict，包含 url 和 title
        - search_and_summarize_webpages: 返回 list，每个元素是 dict，包含 url 和 title
        - search_volc: 返回 list，每个元素是 dict，包含 url 和 title
        - search_tavily: 返回 list，每个元素是 dict，包含 url 和 title

        Args:
            result: 工具执行结果
            tool_name: 工具名称

        Returns:
            (提取到的URL列表, 修改后的result)
        """
        urls = []
        modified_result = result  # 默认返回原始result

        try:
            # 根据工具名称确定引用类型和处理方式
            if tool_name == 'search_wiki':
                # search_wiki 返回字符串，需要解析为字典
                if isinstance(result, str):
                    try:
                        import ast
                        result_dict = ast.literal_eval(result)
                        if isinstance(result_dict, dict) and 'url' in result_dict:
                            url = result_dict['url']
                            title = result_dict.get('title', '')
                            if url and isinstance(url, str):
                                tag = self.reference_manager.add_url(url, title=title, ref_type='webpage')
                                urls.append(url)
                                # 将tag添加到result_dict中
                                result_dict['tag'] = tag
                                # 将修改后的字典作为新的result
                                modified_result = result_dict
                                self._logger.info(f"添加URL到参考文献: {tag} -> {url} (webpage)")
                    except Exception as e:
                        self._logger.warning(f"解析search_wiki结果失败: {e}")
                elif isinstance(result, dict) and 'url' in result:
                    # 如果已经是字典格式
                    url = result['url']
                    title = result.get('title', '')
                    if url and isinstance(url, str):
                        tag = self.reference_manager.add_url(url, title=title, ref_type='webpage')
                        urls.append(url)
                        # 将tag添加到result中（直接修改）
                        result['tag'] = tag
                        modified_result = result
                        self._logger.info(f"添加URL到参考文献: {tag} -> {url} (webpage)")

            elif tool_name == 'search_and_summarize_papers':
                # search_and_summarize_papers 返回 list
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict):
                            # 优先使用 pdf_url，如果没有则尝试 url
                            url = item.get('pdf_url') or item.get('url')
                            title = item.get('title', '')
                            if url and isinstance(url, str):
                                tag = self.reference_manager.add_url(url, title=title, ref_type='paper')
                                urls.append(url)
                                # 将tag添加到item中（直接修改list中的item）
                                item['tag'] = tag
                                self._logger.info(f"添加URL到参考文献: {tag} -> {url} (paper)")
                    modified_result = result  # list已经被直接修改

            elif tool_name in ('search_and_summarize_webpages', 'search_volc', 'search_tavily', 'search_google'):
                # 这些工具都返回 list，每个元素是 dict，包含 url 和 title
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and 'url' in item:
                            url = item['url']
                            title = item.get('title', '')
                            if url and isinstance(url, str) and url.strip():
                                tag = self.reference_manager.add_url(url, title=title, ref_type='webpage')
                                urls.append(url)
                                # 将tag添加到item中（直接修改list中的item）
                                item['tag'] = tag
                                self._logger.info(f"添加URL到参考文献: {tag} -> {url} (webpage)")
                    modified_result = result  # list已经被直接修改

            elif tool_name in ('search_arxiv', 'search_openalex', 'search_crossref'):
                # 论文搜索工具：返回 list，每个元素包含 url/pdf_url 和 title
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict):
                            url = item.get('pdf_url') or item.get('url')
                            title = item.get('title', '')
                            if url and isinstance(url, str) and url.strip():
                                tag = self.reference_manager.add_url(url, title=title, ref_type='paper')
                                urls.append(url)
                                item['tag'] = tag
                                self._logger.info(f"添加URL到参考文献: {tag} -> {url} (paper)")
                    modified_result = result
        
        except Exception as e:
            self._logger.warning(f"从工具结果中提取URL时出错: {e}")
        
        return urls, modified_result
    
    def _append_references_to_text(self, text: str) -> str:
        """
        在文本末尾附加参考文献列表（包含标题和类型）
        
        Args:
            text: 原始文本
            
        Returns:
            附加了参考文献的文本
        """
        # 提取文本中使用的引用标签
        used_tags = set(re.findall(r'\[R(\d+)\]', text))
        
        if not used_tags:
            # 如果没有使用引用标签，不添加参考文献
            return text
        
        # 获取所有参考文献
        all_refs = self.reference_manager.get_reference_list()
        
        # 只包含被使用的参考文献
        used_refs = []
        for ref in all_refs:
            tag_num = ref['tag'][1:]  # 去掉'R'前缀
            if tag_num in used_tags:
                used_refs.append(ref)
        
        if not used_refs:
            return text
        
        # 构建参考文献部分
        ref_lines = ["\n\n## References"]
        for ref in used_refs:
            # 添加类型标记
            type_label = "📄" if ref['type'] == "paper" else "🌐"
            title = ref.get('title', '')
            url = ref['url']
            
            # 格式: [R1] 📄 Title - URL
            if title and title != url:
                ref_lines.append(f"[{ref['tag']}] {type_label} {title} - {url}")
            else:
                ref_lines.append(f"[{ref['tag']}] {type_label} {url}")
        
        return text + "\n".join(ref_lines)
    
    def get_reference_manager(self) -> ReferenceManager:
        """
        获取参考文献管理器
        
        Returns:
            参考文献管理器实例
        """
        return self.reference_manager
