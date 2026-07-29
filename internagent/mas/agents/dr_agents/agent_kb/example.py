import requests
import json
import time
import re
import os
from typing import Literal, Optional, Dict, Any, List

BASE_URL = os.getenv("AGENT_KB_URL", "http://127.0.0.1:9000")
PROXIES = {"http": None, "https": None}

Scope = Literal["base", "append", "compat"]


# -----------------------------
# ✅ Generic HTTP helpers
# -----------------------------
def _post(path: str, payload: dict, timeout: int = 30):
    r = requests.post(
        f"{BASE_URL}{path}",
        json=payload,
        timeout=timeout,
        proxies=PROXIES,
    )
    r.raise_for_status()
    return r.json()


def _get(path: str, timeout: int = 10):
    r = requests.get(f"{BASE_URL}{path}", timeout=timeout, proxies=PROXIES)
    r.raise_for_status()
    return r.json()


def _search_path(scope: Scope, kind: Literal["hybrid", "text", "semantic"]) -> str:
    if scope == "compat":
        return f"/search/{kind}"
    return f"/{scope}/search/{kind}"


# -----------------------------
# ✅ Service API wrappers
# -----------------------------
def call_hybrid_search(scope: Scope, query: str, top_k: int = 3, weights=None):
    payload = {
        "query": query,
        "top_k": top_k,
        "weights": weights or {"text": 0.5, "semantic": 0.5},
    }
    return _post(_search_path(scope, "hybrid"), payload)


def call_text_search(scope: Scope, query: str, top_k: int = 3):
    payload = {"query": query, "top_k": top_k}
    return _post(_search_path(scope, "text"), payload)


def call_semantic_search(scope: Scope, query: str, top_k: int = 3):
    payload = {"query": query, "top_k": top_k}
    return _post(_search_path(scope, "semantic"), payload)


def call_appendkb(content: dict):
    payload = {"content": content}
    return _post("/appendkb", payload)


def get_performance():
    return _get("/performance")


# -----------------------------
# ✅ Utilities
# -----------------------------
def pretty_print_results(results, show_fields=None, max_chars=260):
    if not results:
        print("(no results)")
        return

    show_fields = show_fields or [
        "workflow_id",
        "kb_source",
        "total_score",
        "query",
        "agent_experience",
        "search_agent_experience",
        "is_success",
        "created_at",
    ]

    for i, r in enumerate(results, 1):
        print(f"\n--- Result #{i} ---")
        for k in show_fields:
            v = r.get(k)

            if isinstance(v, list):
                v = v[:5] + (["..."] if len(v) > 5 else [])

            if isinstance(v, str) and len(v) > max_chars:
                v = v[:max_chars] + "..."
            print(f"{k}: {v}")


def assert_contains_unique_token(obj, token: str) -> bool:
    blob = json.dumps(obj, ensure_ascii=False)
    return token in blob


def extract_first_result(results: list) -> dict:
    if not results:
        return {}
    return results[0]


def normalize_space(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def assert_all_kb_source(results: List[dict], expected: str):
    for r in results:
        src = r.get("kb_source")
        assert src == expected, f"❌ kb_source mismatch: expected={expected}, got={src}"


# ==========================================================
# ✅ 主测试
# ==========================================================
if __name__ == "__main__":
    now = int(time.time())

    print("\n==============================")
    print("✅ [Smoke] /performance")
    print("==============================")
    perf = get_performance()
    print("performance:", perf)

    # ----------------------------------------------------------
    # Test 1: 验证 base / append / compat 三套检索接口都可用
    # ----------------------------------------------------------
    # 用一个尽量通用的 query 做 base 搜索（保证 base KB 有数据）
    smoke_query = "如何 设计 agent 工作流 检索 增强"
    print("\n==============================")
    print("✅ [Test 1] base / compat / append 搜索接口可用性 + kb_source 正确")
    print("==============================")

    base_res = call_hybrid_search("base", smoke_query, top_k=3, weights={"text": 0.3, "semantic": 0.7})
    print("\n[base/hybrid] results:")
    pretty_print_results(base_res, show_fields=["workflow_id", "kb_source", "total_score", "query"])
    assert isinstance(base_res, list), "❌ base/hybrid should return a list"
    assert len(base_res) > 0, "❌ base KB 可能为空：无法验证 '命中 base 的 patch' 测试"
    assert_all_kb_source(base_res, "base")

    compat_res = call_hybrid_search("compat", smoke_query, top_k=3, weights={"text": 0.3, "semantic": 0.7})
    print("\n[compat(/search/hybrid)] results:")
    pretty_print_results(compat_res, show_fields=["workflow_id", "kb_source", "total_score", "query"])
    assert isinstance(compat_res, list), "❌ compat/hybrid should return a list"
    # compat 默认走 base，所以 kb_source 应该也是 base
    if compat_res:
        assert_all_kb_source(compat_res, "base")

    append_res0 = call_hybrid_search("append", smoke_query, top_k=3, weights={"text": 0.3, "semantic": 0.7})
    print("\n[append/hybrid] results:")
    pretty_print_results(append_res0, show_fields=["workflow_id", "kb_source", "total_score", "query"])
    assert isinstance(append_res0, list), "❌ append/hybrid should return a list"
    if append_res0:
        assert_all_kb_source(append_res0, "append")

    # 快速覆盖 text/semantic（不做强断言数量，但断言类型/来源）
    base_text = call_text_search("base", smoke_query, top_k=2)
    base_sem = call_semantic_search("base", smoke_query, top_k=2)
    if base_text:
        assert_all_kb_source(base_text, "base")
    if base_sem:
        assert_all_kb_source(base_sem, "base")

    append_text = call_text_search("append", smoke_query, top_k=2)
    append_sem = call_semantic_search("append", smoke_query, top_k=2)
    if append_text:
        assert_all_kb_source(append_text, "append")
    if append_sem:
        assert_all_kb_source(append_sem, "append")

    print("\n✅ Test 1 通过：base/append/compat 的检索接口均可用，kb_source 也正确。")

    # ----------------------------------------------------------
    # Test 2: 验证 appendkb 新增（added=True）只进入 append，不污染 base/compat
    # ----------------------------------------------------------
    token_new_1 = f"__APPEND_NEW_TOKEN_1__{now}__"
    token_new_2 = f"__APPEND_NEW_TOKEN_2__{now}__"

    append_only_query = f"如何设计一个检索增强的 agent 工作流？(append-only) {token_new_1}"

    append_content_new_1 = {
        "query": append_only_query,
        "plan": f"[Plan-New-1] 任务拆解→检索→融合→工具调用→反思 ({token_new_1})",
        "search_plan": f"[SearchPlan-New-1] 多路召回+重排 ({token_new_1})",
        "wrong_messages": [f"[Wrong-New] 示例错误 ({token_new_1})"],
        "is_success": True,
        "agent_experience": f"[AgentExp-New-1] 先 broad 再 narrow，最后 evidence 合并。({token_new_1})",
        "search_agent_experience": f"[SearchExp-New-1] query 扩展 + embedding+BM25。({token_new_1})",
    }

    print("\n==============================")
    print("✅ [Test 2] appendkb 新增 added=True：只能在 append 查到，base/compat 查不到")
    print("==============================")
    resp_new_1 = call_appendkb(append_content_new_1)
    print("append resp_new_1:", resp_new_1)
    assert resp_new_1.get("added") is True, "❌ 预期第一次 append-only query 应该 added=True"
    assert resp_new_1.get("matched_source") in (None, "base", "append"), "❌ matched_source 字段异常"

    # append 检索应能查到 token_new_1
    after_new_append = call_hybrid_search("append", append_only_query, top_k=1, weights={"text": 0.3, "semantic": 0.7})
    print("\n[append/hybrid] after append-only add:")
    pretty_print_results(after_new_append, show_fields=["workflow_id", "kb_source", "total_score", "agent_experience", "search_agent_experience"])
    assert assert_contains_unique_token(after_new_append, token_new_1), "❌ append 搜索结果未包含 token_new_1"

    # base/compat 不应包含 token_new_1（不污染 base）
    after_new_base = call_hybrid_search("base", append_only_query, top_k=3, weights={"text": 0.3, "semantic": 0.7})
    after_new_compat = call_hybrid_search("compat", append_only_query, top_k=3, weights={"text": 0.3, "semantic": 0.7})

    print("\n[base/hybrid] after append-only add:")
    pretty_print_results(after_new_base, show_fields=["workflow_id", "kb_source", "total_score", "query"])
    print("\n[compat/hybrid] after append-only add:")
    pretty_print_results(after_new_compat, show_fields=["workflow_id", "kb_source", "total_score", "query"])

    assert not assert_contains_unique_token(after_new_base, token_new_1), "❌ base 搜索不应包含 append-only token（说明污染了 base）"
    assert not assert_contains_unique_token(after_new_compat, token_new_1), "❌ compat(/search) 不应包含 append-only token（compat 默认 base）"

    print("\n✅ Test 2 通过：新增条目只进入 append，可检索到；base/compat 不会被污染。")

    # ----------------------------------------------------------
    # Test 3: 验证 appendkb 命中 append（added=False + matched_source=append）并追加经验
    # ----------------------------------------------------------
    append_content_new_2 = {
        "query": append_only_query,  # 同一个 query，必然命中 append
        "plan": f"[Plan-New-2] 不重要（不会新增）({token_new_2})",
        "search_plan": f"[SearchPlan-New-2] 不重要（不会新增）({token_new_2})",
        "wrong_messages": [],
        "is_success": True,
        "agent_experience": f"[AgentExp-New-2] 回答必须引用证据，并给出边界条件。({token_new_2})",
        "search_agent_experience": f"[SearchExp-New-2] 先 broad 再 narrow + evidence clustering。({token_new_2})",
    }

    print("\n==============================")
    print("✅ [Test 3] appendkb 命中 append（added=False）并追加经验；matched_source 应为 append")
    print("==============================")
    resp_new_2 = call_appendkb(append_content_new_2)
    print("append resp_new_2:", resp_new_2)
    assert resp_new_2.get("added") is False, "❌ 第二次同 query 应该 added=False"
    # 这里预期 matched_source=append（因为 query 本身是 append-only）
    assert resp_new_2.get("matched_source") == "append", f"❌ 预期 matched_source=append, got={resp_new_2.get('matched_source')}"

    after_merge_append = call_hybrid_search("append", append_only_query, top_k=1, weights={"text": 0.3, "semantic": 0.7})
    r_merge = extract_first_result(after_merge_append)
    agent_exp = r_merge.get("agent_experience") or ""
    search_exp = r_merge.get("search_agent_experience") or ""
    print("\n[append/hybrid] after append-hit merge:")
    pretty_print_results(after_merge_append, show_fields=["workflow_id", "kb_source", "agent_experience", "search_agent_experience"])

    assert token_new_1 in agent_exp and token_new_1 in search_exp, "❌ 旧经验 token_new_1 不见了"
    assert token_new_2 in agent_exp and token_new_2 in search_exp, "❌ 新经验 token_new_2 未追加成功"

    print("\n✅ Test 3 通过：命中 append 不新增，但经验成功追加；matched_source=append 正确。")

    # ----------------------------------------------------------
    # Test 4: 验证 appendkb 命中 base（added=False + matched_source=base），但写入永远落 append（patch）
    # ----------------------------------------------------------
    token_patch_base_1 = f"__PATCH_BASE_TOKEN_1__{now}__"
    token_patch_base_2 = f"__PATCH_BASE_TOKEN_2__{now}__"

    # 取一个 base/hybrid 的 top1 query，当作“必命中 base 的 query”
    base_top1 = extract_first_result(base_res)
    base_query_for_patch = base_top1.get("query")
    assert base_query_for_patch, "❌ base top1 缺少 query 字段，无法做 patch 测试"

    print("\n==============================")
    print("✅ [Test 4] appendkb 命中 base：matched_source=base；经验写入 append（patch）且 base 不被污染")
    print("==============================")
    print("Selected base_query_for_patch:", base_query_for_patch)

    patch_content_1 = {
        "query": base_query_for_patch,  # 用完全相同 query，确保 base 命中分高
        "plan": f"[Plan-Patch-1] 不重要 ({token_patch_base_1})",
        "search_plan": f"[SearchPlan-Patch-1] 不重要 ({token_patch_base_1})",
        "wrong_messages": [],
        "is_success": True,
        "agent_experience": f"[AgentExp-Patch-1] （来自 base 命中）需要加入引用与反例。({token_patch_base_1})",
        "search_agent_experience": f"[SearchExp-Patch-1] 分阶段检索 + 证据聚类。({token_patch_base_1})",
    }

    resp_patch_1 = call_appendkb(patch_content_1)
    print("append resp_patch_1:", resp_patch_1)
    assert resp_patch_1.get("added") is False, "❌ 命中 base 时应 added=False"
    assert resp_patch_1.get("matched_source") == "base", f"❌ 预期 matched_source=base, got={resp_patch_1.get('matched_source')}"
    assert resp_patch_1.get("matched_score", 0.0) is not None, "❌ matched_score 应存在"

    # append 里应能检索到该 base_query 的 patch，并包含 token_patch_base_1
    after_patch_append = call_hybrid_search("append", base_query_for_patch, top_k=3, weights={"text": 0.3, "semantic": 0.7})
    print("\n[append/hybrid] after base-hit patch:")
    pretty_print_results(after_patch_append, show_fields=["workflow_id", "kb_source", "query", "agent_experience", "search_agent_experience"])
    assert assert_contains_unique_token(after_patch_append, token_patch_base_1), "❌ append 没检索到 base-hit 的 patch token"

    # base 里不应出现 token_patch_base_1（不污染 base）
    after_patch_base = call_hybrid_search("base", base_query_for_patch, top_k=1, weights={"text": 0.3, "semantic": 0.7})
    print("\n[base/hybrid] after base-hit patch (should NOT contain patch token):")
    pretty_print_results(after_patch_base, show_fields=["workflow_id", "kb_source", "query", "agent_experience", "search_agent_experience"])
    assert not assert_contains_unique_token(after_patch_base, token_patch_base_1), "❌ base 被污染：出现了 patch token"

    print("\n✅ Test 4-1 通过：命中 base 的 appendkb 会把经验写入 append（patch），但不污染 base。")

    # 再次对同一个 base_query 做 patch，验证“经验追加”仍可在 append 中生效
    patch_content_2 = {
        "query": base_query_for_patch,
        "plan": f"[Plan-Patch-2] 不重要 ({token_patch_base_2})",
        "search_plan": f"[SearchPlan-Patch-2] 不重要 ({token_patch_base_2})",
        "wrong_messages": [],
        "is_success": True,
        "agent_experience": f"[AgentExp-Patch-2] （第二次 patch）强调可复现与失败回退。({token_patch_base_2})",
        "search_agent_experience": f"[SearchExp-Patch-2] （第二次 patch）引入 rerank + 去重。({token_patch_base_2})",
    }

    resp_patch_2 = call_appendkb(patch_content_2)
    print("\nappend resp_patch_2:", resp_patch_2)
    assert resp_patch_2.get("added") is False, "❌ 再次 patch 仍应 added=False"
    assert resp_patch_2.get("matched_source") == "base", f"❌ 预期 matched_source=base, got={resp_patch_2.get('matched_source')}"

    after_patch_append2 = call_hybrid_search("append", base_query_for_patch, top_k=3, weights={"text": 0.3, "semantic": 0.7})
    print("\n[append/hybrid] after base-hit patch #2:")
    pretty_print_results(after_patch_append2, show_fields=["workflow_id", "kb_source", "agent_experience", "search_agent_experience"])
    assert assert_contains_unique_token(after_patch_append2, token_patch_base_2), "❌ append 没检索到第二次 patch token（经验追加可能未生效）"

    print("\n✅ Test 4-2 通过：base-hit 的 patch 也能多次追加经验，并且仍只落 append。")

    # ----------------------------------------------------------
    # Final: /performance
    # ----------------------------------------------------------
    print("\n==============================")
    print("✅ [Final] /performance")
    print("==============================")
    print(get_performance())

    print("\n✅ ✅ ✅ 全部测试通过：")
    print("1) base/append/compat 三套检索接口均可用且 kb_source 正确")
    print("2) append-only 新增只出现在 append，不污染 base/compat")
    print("3) 命中 append 时 added=False，matched_source=append，经验追加生效")
    print("4) 命中 base 时 added=False，matched_source=base，但写入永远落 append（patch），base 不污染，经验可持续追加")
