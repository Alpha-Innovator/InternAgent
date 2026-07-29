from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import json
import asyncio
import tempfile
from datetime import datetime

from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer

from agent_kb_retrieval import AKB_Manager, WorkflowInstance

MAX_CONCURRENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_SEARCHES", 10))
CACHE_TTL = 60
SIM_THRESHOLD = float(os.getenv("APPENDKB_SIM_THRESHOLD", 0.8))

BASE_KB_PATH = os.getenv("BASE_KB_PATH", "./agent_kb/agent_kb_database.json")
APPEND_KB_PATH = os.getenv("APPEND_KB_PATH", "./agent_kb/agent_kb_append.json")

app = FastAPI(title="Optimized Knowledge Retrieval API (Base + Append Split)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 经验拼接分隔符（可改）
EXP_SEP = "\n\n---\n\n"


def ensure_json_list_file(path: str):
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError("append kb file must be a JSON list")
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def atomic_write_json(path: str, data: list):
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="appendkb_", suffix=".json", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


# -----------------------------
# ✅ 经验相似度 + 合并逻辑（lex + semantic）
# -----------------------------
def lexical_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    return float(np.dot(v1, v2) / denom)


def semantic_similarity(embedding_model: SentenceTransformer, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    emb_a = embedding_model.encode(a, convert_to_numpy=True)
    emb_b = embedding_model.encode(b, convert_to_numpy=True)
    return cosine_sim(emb_a, emb_b)


def hybrid_similarity(embedding_model: SentenceTransformer, a: str, b: str, alpha: float = 0.5) -> float:
    sem = semantic_similarity(embedding_model, a, b)
    lex = lexical_similarity(a, b)
    return alpha * sem + (1 - alpha) * lex


def normalize_experiences(exp: Optional[str]) -> List[str]:
    if not exp:
        return []
    parts = [p.strip() for p in exp.split(EXP_SEP)]
    return [p for p in parts if p]


def merge_experience_if_needed(
    embedding_model: SentenceTransformer,
    old_exp: Optional[str],
    new_exp: Optional[str],
    sim_threshold: float = SIM_THRESHOLD,
    alpha: float = 0.5,
) -> Tuple[Optional[str], bool, float]:
    """
    如果 new_exp 与 old_exp 中任意片段最大相似度 < threshold，则追加；
    返回 (merged_exp, changed, best_sim)
    """
    new_exp = (new_exp or "").strip()
    if not new_exp:
        return old_exp, False, 1.0

    old_list = normalize_experiences(old_exp)
    if not old_list:
        return new_exp, True, 0.0

    best_sim = 0.0
    for old_item in old_list:
        sim = hybrid_similarity(embedding_model, old_item, new_exp, alpha=alpha)
        best_sim = max(best_sim, sim)

    if best_sim < sim_threshold:
        merged = (old_exp or "").strip()
        if merged:
            merged = merged + EXP_SEP + new_exp
        else:
            merged = new_exp
        return merged, True, best_sim

    return old_exp, False, best_sim


# -----------------------------
# ✅ 初始化
# -----------------------------
ensure_json_list_file(APPEND_KB_PATH)

# ✅ 两个 KB 独立 manager（共享同一个 embedding_model，省内存）
shared_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

base_manager = AKB_Manager(
    json_file_paths=[BASE_KB_PATH],
    kb_source="base",
    shared_embedding_model=shared_model,
)
append_manager = AKB_Manager(
    json_file_paths=[APPEND_KB_PATH],
    kb_source="append",
    shared_embedding_model=shared_model,
)

performance_stats = {
    "total_requests": 0,
    "avg_response_time": 0.0,
    "last_updated": time.time(),
}

response_cache: Dict[str, Dict[str, Any]] = {}
append_lock = asyncio.Lock()


def update_performance_stats(response_time: float):
    total_time = performance_stats["avg_response_time"] * performance_stats["total_requests"]
    performance_stats["total_requests"] += 1
    performance_stats["avg_response_time"] = (total_time + response_time) / performance_stats["total_requests"]
    performance_stats["last_updated"] = time.time()


# -----------------------------
# ✅ Pydantic Models
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 1
    weights: Optional[Dict[str, float]] = {"text": 0.5, "semantic": 0.5}


class WorkflowResponse(BaseModel):
    workflow_id: str
    total_score: Optional[float] = None
    kb_source: Optional[str] = None

    query: str
    plan: Optional[str] = None
    search_plan: Optional[str] = None
    wrong_messages: Optional[List[str]] = None
    agent_experience: Optional[str] = None
    search_agent_experience: Optional[str] = None
    is_success: Optional[bool] = None
    created_at: Optional[str] = None


class PerformanceStats(BaseModel):
    total_requests: int
    avg_response_time: float
    cache_hit_rate: float


class AppendKBContent(BaseModel):
    query: str
    plan: Optional[str] = None
    search_plan: Optional[str] = None
    wrong_messages: Optional[List[str]] = None
    agent_experience: Optional[str] = None
    search_agent_experience: Optional[str] = None
    is_success: Optional[bool] = None


class AppendKBRequest(BaseModel):
    content: AppendKBContent = Field(
        ...,
        description="Keys: query, plan, search_plan, wrong_messages, agent_experience, search_agent_experience, is_success",
    )


class AppendKBResponse(BaseModel):
    added: bool
    workflow_id: Optional[str] = None
    matched_score: Optional[float] = None
    matched_source: Optional[str] = None
    append_kb_size: int


# -----------------------------
# ✅ helper：从 base & append 中选出 query 最相似的 top1
# -----------------------------
def best_query_hit(q: str) -> Tuple[Optional[str], float, Optional[str]]:
    """
    返回 (workflow_id, score, source)
    source ∈ {"base","append",None}
    """
    hit_base = base_manager.knowledge_base.field_semantic_search(q, "query", top_k=1)
    hit_append = append_manager.knowledge_base.field_semantic_search(q, "query", top_k=1)

    best_id, best_score, best_source = None, 0.0, None

    if hit_base:
        s = float(hit_base[0].get("score", 0.0))
        if s > best_score:
            best_score = s
            best_id = hit_base[0].get("workflow_id")
            best_source = "base"

    if hit_append:
        s = float(hit_append[0].get("score", 0.0))
        if s > best_score:
            best_score = s
            best_id = hit_append[0].get("workflow_id")
            best_source = "append"

    return best_id, best_score, best_source


def read_append_json() -> List[dict]:
    with open(APPEND_KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def upsert_append_patch(
    append_data: List[dict],
    wf_query: str,
    agent_planning: Optional[str],
    search_agent_planning: Optional[str],
    wrong_messages: Optional[List[str]],
    agent_experience: Optional[str],
    search_agent_experience: Optional[str],
    is_success: Optional[bool],
    created_at: Optional[str],
) -> bool:
    """
    在 append json 中：
    - 如果存在 question==wf_query：更新经验字段
    - 否则 append 一条 patch 记录
    返回是否发生变更（True=写入/更新了）
    """
    updated = False
    for item in append_data:
        if (item.get("question") or "").strip() == (wf_query or "").strip():
            # 只更新经验字段（也可以按需更新其他字段）
            item["agent_experience"] = agent_experience
            item["search_agent_experience"] = search_agent_experience
            updated = True
            break

    if not updated:
        append_data.append(
            {
                "question": wf_query,
                "agent_planning": agent_planning,
                "search_agent_planning": search_agent_planning,
                "wrong_messages": wrong_messages or [],
                "agent_experience": agent_experience,
                "search_agent_experience": search_agent_experience,
                "is_success": is_success,
                "created_at": created_at or datetime.utcnow().isoformat(),
            }
        )
        return True

    return True


# -----------------------------
# ✅ appendkb（只写 append，但相似命中可来自 base 或 append）
# -----------------------------
@app.post("/appendkb", response_model=AppendKBResponse)
async def appendkb(request: AppendKBRequest):
    """
    1) 对 base + append 做 query 语义检索，取最佳 top1
       - 若 top1 score >= SIM_THRESHOLD：
         不新增“新 query”，但会把经验合并后写入 append（patch 或更新现有 append 条目）
         并同步更新 append_manager 内存索引（必要时重建 tfidf）
    2) 否则：新增到 append json + append_manager.append_kb_entry + 清缓存
    """
    async with append_lock:
        try:
            content = request.content
            q = (content.query or "").strip()
            if not q:
                raise HTTPException(status_code=400, detail="content.query is required")

            append_data = read_append_json()

            # --- Step A: query 相似检索（base & append 都查，取更高者） ---
            best_wf_id, best_score, best_source = best_query_hit(q)

            if best_wf_id and best_score >= SIM_THRESHOLD and best_source in ("base", "append"):
                # 命中的 workflow（来自 base 或 append）
                if best_source == "base":
                    wf = base_manager.get_workflow_details(best_wf_id)
                else:
                    wf = append_manager.get_workflow_details(best_wf_id)

                # 即使命中 id 找不到，也要返回 matched 信息
                if not wf:
                    return AppendKBResponse(
                        added=False,
                        workflow_id=best_wf_id,
                        matched_score=float(best_score),
                        matched_source=best_source,
                        append_kb_size=len(append_data),
                    )

                # --- Step A1: 合并经验（基于命中 wf 的旧经验 + 新经验）---
                merged_agent_exp, agent_changed, _ = merge_experience_if_needed(
                    shared_model,
                    wf.agent_experience,
                    content.agent_experience,
                    sim_threshold=SIM_THRESHOLD,
                )

                merged_search_agent_exp, search_changed, _ = merge_experience_if_needed(
                    shared_model,
                    wf.search_agent_experience,
                    content.search_agent_experience,
                    sim_threshold=SIM_THRESHOLD,
                )

                # --- Step A2: 如果经验需要更新 → 写入 append（update 或 patch），并更新 append_manager 内存 ---
                if agent_changed or search_changed:
                    wf_query = (wf.query or "").strip()

                    # ✅ 1) 先 upsert 到 append json（无论命中来自 base 还是 append，都只写 append）
                    upsert_append_patch(
                        append_data=append_data,
                        wf_query=wf_query,
                        agent_planning=wf.agent_planning,
                        search_agent_planning=wf.search_agent_planning,
                        wrong_messages=wf.wrong_messages or [],
                        agent_experience=merged_agent_exp,
                        search_agent_experience=merged_search_agent_exp,
                        is_success=wf.is_success,
                        created_at=wf.created_at or datetime.utcnow().isoformat(),
                    )
                    atomic_write_json(APPEND_KB_PATH, append_data)

                    # ✅ 2) 更新 append_manager 内存（两种情况）
                    if best_source == "append":
                        # 命中本就在 append：直接更新该 workflow 的经验字段
                        wf.agent_experience = merged_agent_exp
                        wf.search_agent_experience = merged_search_agent_exp
                        # TF-IDF 不必一定 refit（query 不变），但为安全起见可 refit
                        append_manager.knowledge_base.build_tfidf_indices()
                        # embedding 不变（query 不变），不需要重建
                    else:
                        # 命中来自 base：append 里可能没有对应条目，需要确保内存里也有一条 patch/workflow
                        # 做一个“补写入内存”的 upsert：如果 append 内存里已存在同 query，更新；否则新增
                        existing_id = None
                        for wid, w in append_manager.knowledge_base.workflows.items():
                            if (w.query or "").strip() == wf_query:
                                existing_id = wid
                                break

                        if existing_id:
                            w = append_manager.knowledge_base.workflows[existing_id]
                            w.agent_experience = merged_agent_exp
                            w.search_agent_experience = merged_search_agent_exp
                            append_manager.knowledge_base.build_tfidf_indices()
                        else:
                            # 新增一个 patch workflow 到 append 内存（并 refit tfidf）
                            append_manager.append_kb_entry(
                                {
                                    "query": wf_query,
                                    "plan": wf.agent_planning,
                                    "search_plan": wf.search_agent_planning,
                                    "wrong_messages": wf.wrong_messages or [],
                                    "agent_experience": merged_agent_exp,
                                    "search_agent_experience": merged_search_agent_exp,
                                    "is_success": wf.is_success,
                                    "created_at": wf.created_at or datetime.utcnow().isoformat(),
                                }
                            )

                    # ✅ 3) 清缓存（因为 append 检索结果会变）
                    response_cache.clear()

                return AppendKBResponse(
                    added=False,
                    workflow_id=best_wf_id,
                    matched_score=float(best_score),
                    matched_source=best_source,
                    append_kb_size=len(append_data),
                )

            # --- Step B: query 不相似 → 新增 append 条目 ---
            created_at = datetime.utcnow().isoformat()
            new_item = {
                "question": q,
                "agent_planning": content.plan,
                "search_agent_planning": content.search_plan,
                "wrong_messages": content.wrong_messages or [],
                "agent_experience": content.agent_experience,
                "search_agent_experience": content.search_agent_experience,
                "is_success": content.is_success,
                "created_at": created_at,
            }

            append_data.append(new_item)
            atomic_write_json(APPEND_KB_PATH, append_data)

            # --- Step C: 更新 append_manager 内存索引 ---
            wf_new = append_manager.append_kb_entry(
                {
                    "query": new_item["question"],
                    "plan": new_item.get("agent_planning"),
                    "search_plan": new_item.get("search_agent_planning"),
                    "wrong_messages": new_item.get("wrong_messages"),
                    "agent_experience": new_item.get("agent_experience"),
                    "search_agent_experience": new_item.get("search_agent_experience"),
                    "is_success": new_item.get("is_success"),
                    "created_at": new_item.get("created_at"),
                }
            )

            response_cache.clear()

            return AppendKBResponse(
                added=True,
                workflow_id=wf_new.workflow_id,
                matched_score=None,
                matched_source=None,
                append_kb_size=len(append_data),
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"appendkb failed: {str(e)}")


# -----------------------------
# ✅ 搜索接口：Base KB
# -----------------------------
@app.post("/base/search/hybrid", response_model=List[WorkflowResponse])
async def base_hybrid_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"base_hybrid_{request.query}_{request.top_k}_{request.weights}"

    try:
        if cache_key in response_cache and time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
            return response_cache[cache_key]["data"]

        results = base_manager.hybrid_search(query=request.query, top_k=request.top_k, weights=request.weights)
        response_data = [WorkflowResponse(**item) for item in results]

        response_cache[cache_key] = {"timestamp": time.time(), "data": response_data}
        update_performance_stats(time.time() - start_time)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base hybrid search failed: {str(e)}")


@app.post("/base/search/text", response_model=List[WorkflowResponse])
async def base_text_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"base_text_{request.query}_{request.top_k}"

    try:
        if cache_key in response_cache and time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
            return response_cache[cache_key]["data"]

        raw_results = base_manager.search_by_text(request.query, "query", request.top_k)
        response_data = [
            WorkflowResponse(
                workflow_id=item["workflow_id"],
                total_score=item["score"],
                kb_source=item.get("kb_source"),
                **item["content"],
            )
            for item in raw_results
        ]

        response_cache[cache_key] = {"timestamp": time.time(), "data": response_data}
        update_performance_stats(time.time() - start_time)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base text search failed: {str(e)}")


@app.post("/base/search/semantic", response_model=List[WorkflowResponse])
async def base_semantic_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"base_semantic_{request.query}_{request.top_k}"

    try:
        if cache_key in response_cache and time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
            return response_cache[cache_key]["data"]

        raw_results = base_manager.search_by_semantic(request.query, "query", request.top_k)
        response_data = [
            WorkflowResponse(
                workflow_id=item["workflow_id"],
                total_score=item["score"],
                kb_source=item.get("kb_source"),
                **item["content"],
            )
            for item in raw_results
        ]

        response_cache[cache_key] = {"timestamp": time.time(), "data": response_data}
        update_performance_stats(time.time() - start_time)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base semantic search failed: {str(e)}")


# -----------------------------
# ✅ 搜索接口：Append KB
# -----------------------------
@app.post("/append/search/hybrid", response_model=List[WorkflowResponse])
async def append_hybrid_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"append_hybrid_{request.query}_{request.top_k}_{request.weights}"

    try:
        if cache_key in response_cache and time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
            return response_cache[cache_key]["data"]

        results = append_manager.hybrid_search(query=request.query, top_k=request.top_k, weights=request.weights)
        response_data = [WorkflowResponse(**item) for item in results]

        response_cache[cache_key] = {"timestamp": time.time(), "data": response_data}
        update_performance_stats(time.time() - start_time)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Append hybrid search failed: {str(e)}")


@app.post("/append/search/text", response_model=List[WorkflowResponse])
async def append_text_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"append_text_{request.query}_{request.top_k}"

    try:
        if cache_key in response_cache and time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
            return response_cache[cache_key]["data"]

        raw_results = append_manager.search_by_text(request.query, "query", request.top_k)
        response_data = [
            WorkflowResponse(
                workflow_id=item["workflow_id"],
                total_score=item["score"],
                kb_source=item.get("kb_source"),
                **item["content"],
            )
            for item in raw_results
        ]

        response_cache[cache_key] = {"timestamp": time.time(), "data": response_data}
        update_performance_stats(time.time() - start_time)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Append text search failed: {str(e)}")


@app.post("/append/search/semantic", response_model=List[WorkflowResponse])
async def append_semantic_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"append_semantic_{request.query}_{request.top_k}"

    try:
        if cache_key in response_cache and time.time() - response_cache[cache_key]["timestamp"] < CACHE_TTL:
            return response_cache[cache_key]["data"]

        raw_results = append_manager.search_by_semantic(request.query, "query", request.top_k)
        response_data = [
            WorkflowResponse(
                workflow_id=item["workflow_id"],
                total_score=item["score"],
                kb_source=item.get("kb_source"),
                **item["content"],
            )
            for item in raw_results
        ]

        response_cache[cache_key] = {"timestamp": time.time(), "data": response_data}
        update_performance_stats(time.time() - start_time)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Append semantic search failed: {str(e)}")


# -----------------------------
# ✅ 兼容旧接口（可选）：默认走 base
# -----------------------------
@app.post("/search/hybrid", response_model=List[WorkflowResponse])
async def hybrid_search_compat(request: SearchRequest):
    return await base_hybrid_search(request)


@app.post("/search/text", response_model=List[WorkflowResponse])
async def text_search_compat(request: SearchRequest):
    return await base_text_search(request)


@app.post("/search/semantic", response_model=List[WorkflowResponse])
async def semantic_search_compat(request: SearchRequest):
    return await base_semantic_search(request)


@app.get("/performance", response_model=PerformanceStats)
async def get_performance():
    cache_hit_rate = (
        sum(1 for v in response_cache.values() if time.time() - v["timestamp"] < CACHE_TTL) / len(response_cache)
        if response_cache
        else 0
    )
    return {
        "total_requests": performance_stats["total_requests"],
        "avg_response_time": performance_stats["avg_response_time"],
        "cache_hit_rate": cache_hit_rate,
    }


# -----------------------------
# ✅ 启动
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 9000)),
        workers=int(os.getenv("UVICORN_WORKERS", 1)),
        limit_concurrency=MAX_CONCURRENT_SEARCHES,
    )
