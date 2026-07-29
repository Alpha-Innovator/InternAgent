import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


@dataclass
class WorkflowInstance:
    """A complete workflow instance"""
    workflow_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    query: str = ""
    agent_planning: Optional[str] = None
    search_agent_planning: Optional[str] = None
    wrong_messages: Optional[List[str]] = None
    agent_experience: Optional[str] = None
    search_agent_experience: Optional[str] = None
    is_success: Optional[bool] = None
    created_at: Optional[str] = None

    query_embedding: Optional[np.ndarray] = None
    plan_embedding: Optional[np.ndarray] = None
    search_plan_embedding: Optional[np.ndarray] = None


class AgenticKnowledgeBase:
    def __init__(self, json_file_paths=None, embedding_model: Optional[SentenceTransformer] = None):
        self.workflows: Dict[str, WorkflowInstance] = {}

        # ✅ 允许外部传入共享的 embedding model（节省内存/加载时间）
        self.embedding_model = embedding_model or SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.field_components = {
            "query": {
                "vectorizer": TfidfVectorizer(stop_words="english"),
                "matrix": None,
                "workflow_ids": [],
            }
        }

        if json_file_paths:
            self.load_initial_data(json_file_paths)
            self.finalize_index()

    def load_initial_data(self, json_file_paths):
        for json_path in json_file_paths:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            self.parse_json_file(json_path)

    def parse_json_file(self, json_file_path):
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"Skipping {json_file_path}: JSON root is not a list")
                return

            batch: List[WorkflowInstance] = []
            for item in data:
                try:
                    instance = WorkflowInstance(
                        query=item.get("question", "") or "",
                        agent_planning=item.get("agent_planning"),
                        search_agent_planning=item.get("search_agent_planning"),
                        wrong_messages=item.get("wrong_messages"),
                        agent_experience=item.get("agent_experience"),
                        search_agent_experience=item.get("search_agent_experience"),
                        is_success=item.get("is_success"),
                        created_at=item.get("created_at"),
                    )
                    batch.append(instance)
                except Exception as e:
                    print(f"Skipping invalid item: {e}")
                    continue

            for instance in batch:
                self.workflows[instance.workflow_id] = instance

        except Exception as e:
            print(f"Error parsing file {json_file_path}: {e}")

    def add_workflow_instance(self, workflow: WorkflowInstance):
        self.workflows[workflow.workflow_id] = workflow
        return workflow

    def finalize_index(self):
        print("Building search indices...")
        self.build_tfidf_indices()
        self.build_embeddings()

    def build_tfidf_indices(self):
        """Build TF-IDF indices in batch (refit)"""
        field_data = {"query": []}

        for workflow in self.workflows.values():
            field_data["query"].append(workflow.query)

        if not field_data["query"]:
            return

        vectorizer = self.field_components["query"]["vectorizer"]
        self.field_components["query"]["matrix"] = vectorizer.fit_transform(field_data["query"])
        self.field_components["query"]["workflow_ids"] = list(self.workflows.keys())

    def build_embeddings(self):
        print("Generating embeddings...")
        workflows = list(self.workflows.values())
        if not workflows:
            return

        batch_size = 32
        queries = [w.query for w in workflows]
        query_embeddings = self.embedding_model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        for i, workflow in enumerate(workflows):
            workflow.query_embedding = query_embeddings[i]

    def field_text_search(self, query: str, field: str, top_k: int = 3) -> List[dict]:
        component = self.field_components.get(field)
        if not component or component["matrix"] is None or not component["workflow_ids"]:
            return []

        query_vec = component["vectorizer"].transform([query])
        similarities = cosine_similarity(query_vec, component["matrix"]).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            wf_id = component["workflow_ids"][idx]
            wf = self.workflows[wf_id]
            results.append(
                {
                    "workflow_id": wf_id,
                    "score": float(similarities[idx]),
                    "field": field,
                    "content": wf.query,
                }
            )
        return results

    def field_semantic_search(self, query: str, field: str, top_k: int = 3) -> List[dict]:
        """Optimized semantic search"""
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        embeddings = []
        workflows = []
        for workflow in self.workflows.values():
            if workflow.query_embedding is not None:
                embeddings.append(workflow.query_embedding)
                workflows.append(workflow)

        if not embeddings:
            return []

        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            wf = workflows[idx]
            results.append(
                {
                    "workflow_id": wf.workflow_id,
                    "score": float(similarities[idx]),
                    "field": field,
                    "content": wf.query,
                }
            )
        return results

    def add_item_and_index(
        self,
        *,
        query: str,
        agent_planning: Optional[str] = None,
        search_agent_planning: Optional[str] = None,
        wrong_messages: Optional[List[str]] = None,
        agent_experience: Optional[str] = None,
        search_agent_experience: Optional[str] = None,
        is_success: Optional[bool] = None,
        created_at: Optional[str] = None,
        rebuild_tfidf: bool = True,
    ) -> WorkflowInstance:
        """
        增量添加一条：
        - embedding：只对新增 query 编码
        - TF-IDF：sklearn 不支持真正增量，使用 refit
        """
        instance = WorkflowInstance(
            query=query or "",
            agent_planning=agent_planning,
            search_agent_planning=search_agent_planning,
            wrong_messages=wrong_messages or [],
            agent_experience=agent_experience,
            search_agent_experience=search_agent_experience,
            is_success=is_success,
            created_at=created_at or datetime.utcnow().isoformat(),
        )
        self.workflows[instance.workflow_id] = instance

        instance.query_embedding = self.embedding_model.encode(instance.query, convert_to_numpy=True)

        if rebuild_tfidf:
            self.build_tfidf_indices()

        return instance


class AKB_Manager:
    """
    ✅ 一个 manager 对应一个 KB（base 或 append），索引完全隔离
    """

    def __init__(
        self,
        json_file_paths=None,
        kb_source: str = "unknown",
        shared_embedding_model: Optional[SentenceTransformer] = None,
    ):
        self.kb_source = kb_source
        self.knowledge_base = AgenticKnowledgeBase(
            json_file_paths=json_file_paths,
            embedding_model=shared_embedding_model,
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        weights: Dict[str, float] = None,
    ) -> List[dict]:
        weights = weights or {"text": 0.5, "semantic": 0.5}
        field_weights = {"query": 1.0}

        score_board = defaultdict(float)

        for field in ["query"]:
            for result in self.knowledge_base.field_text_search(query, field, top_k * 2):
                score_board[result["workflow_id"]] += weights["text"] * field_weights[field] * result["score"]
            for result in self.knowledge_base.field_semantic_search(query, field, top_k * 2):
                score_board[result["workflow_id"]] += weights["semantic"] * field_weights[field] * result["score"]

        sorted_results = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[:top_k]

        detailed_results = []
        for wf_id, total_score in sorted_results:
            workflow = self.knowledge_base.workflows[wf_id]
            detailed_results.append(
                {
                    "workflow_id": wf_id,
                    "total_score": float(total_score),
                    "kb_source": self.kb_source,
                    "query": workflow.query,
                    "plan": workflow.agent_planning,
                    "search_plan": workflow.search_agent_planning,
                    "wrong_messages": workflow.wrong_messages,
                    "agent_experience": workflow.agent_experience,
                    "search_agent_experience": workflow.search_agent_experience,
                    "is_success": workflow.is_success,
                    "created_at": workflow.created_at,
                }
            )

        return detailed_results

    def search_by_text(self, query: str, field: str = "query", top_k: int = 3) -> List[dict]:
        results = []
        for result in self.knowledge_base.field_text_search(query, field, top_k):
            workflow = self.get_workflow_details(result["workflow_id"])
            if not workflow:
                continue
            results.append(
                {
                    "workflow_id": result["workflow_id"],
                    "score": float(result["score"]),
                    "kb_source": self.kb_source,
                    "content": {
                        "query": workflow.query,
                        "plan": workflow.agent_planning,
                        "search_plan": workflow.search_agent_planning,
                        "wrong_messages": workflow.wrong_messages,
                        "agent_experience": workflow.agent_experience,
                        "search_agent_experience": workflow.search_agent_experience,
                        "is_success": workflow.is_success,
                        "created_at": workflow.created_at,
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def search_by_semantic(self, query: str, field: str = "query", top_k: int = 3) -> List[dict]:
        results = []
        for result in self.knowledge_base.field_semantic_search(query, field, top_k):
            workflow = self.get_workflow_details(result["workflow_id"])
            if not workflow:
                continue
            results.append(
                {
                    "workflow_id": result["workflow_id"],
                    "score": float(result["score"]),
                    "kb_source": self.kb_source,
                    "content": {
                        "query": workflow.query,
                        "plan": workflow.agent_planning,
                        "search_plan": workflow.search_agent_planning,
                        "wrong_messages": workflow.wrong_messages,
                        "agent_experience": workflow.agent_experience,
                        "search_agent_experience": workflow.search_agent_experience,
                        "is_success": workflow.is_success,
                        "created_at": workflow.created_at,
                    },
                }
            )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def get_workflow_details(self, workflow_id: str) -> Optional[WorkflowInstance]:
        return self.knowledge_base.workflows.get(workflow_id)

    def append_kb_entry(self, content: Dict[str, Any]) -> WorkflowInstance:
        """
        ✅ 仅用于 append KB manager：把新增条目写入内存并建索引
        """
        return self.knowledge_base.add_item_and_index(
            query=content.get("query", "") or "",
            agent_planning=content.get("plan"),
            search_agent_planning=content.get("search_plan"),
            wrong_messages=content.get("wrong_messages") or [],
            agent_experience=content.get("agent_experience"),
            search_agent_experience=content.get("search_agent_experience"),
            is_success=content.get("is_success"),
            created_at=content.get("created_at"),
            rebuild_tfidf=True,
        )
