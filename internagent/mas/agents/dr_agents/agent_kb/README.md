# agent_kb — Strategy-Procedural Memory (SPM) for Deep Research

This package is the **Strategy-Procedural Memory (SPM)** described in the paper
(Section 2.4.1), wired into the Deep Research path (`dr_agents`).

Each completed task distills into two kinds of experience, stored in a persistent
experience knowledge base and retrieved before executing similar future subtasks:

| SPM concept | field | prompt label |
|---|---|---|
| **Strategy memory** (planning) | `agent_experience` | `PLANNING EXPERIENCE` |
| **Procedural memory** (execution) | `search_agent_experience` | `EXECUTION EXPERIENCE` |

## How it is wired into `dr_agents`

- **Retrieval** — `agents/task/execution_agent.py` calls `call_hybrid_search('base'/'append', task)`
  before a subtask and injects the results into `EXECUTION_PROMPT_WITHMEMORY`.
- **Write-back** — `workflow/task.py` collects the subtask trace, distills it with
  `MEMORY_REASONING_PROMPT`, and calls `call_appendkb(...)` after the task finishes.

Both are **gated by the `need_memory` env var** and **degrade gracefully**: if this
package or the service is unavailable, the DR agents fall back to the normal
(non-memory) prompts and skip write-back. SPM is therefore **off by default**.

## Files

- `example.py` — HTTP client used by the DR agents at runtime (only needs `requests`).
- `memory_utils.py` — `remove_tool_messages`, `extract_agent_and_search_experience`.
- `agent_kb_service.py` — standalone FastAPI service (base + append KB, hybrid retrieval).
- `agent_kb_retrieval.py`, `agent_kb_utils.py`, `kb_prompt.py` — service internals.
- `prompts.yaml` — service-side prompts.
- `requirements.txt` — dependencies for running the **service** (fastapi, uvicorn,
  sentence-transformers, scikit-learn, numpy, …). The client (`example.py`) itself
  only needs `requests`.

> Note: the large KB **data** files (`agent_kb_database.json`, `agent_kb_append.json`)
> are intentionally **not** bundled. Point the service at them via env vars.

## Running the SPM service

```bash
cd internagent/mas/agents/dr_agents/agent_kb
pip install -r requirements.txt

# Point at your KB data (defaults: ./agent_kb_database.json / ./agent_kb_append.json)
export BASE_KB_PATH=/path/to/agent_kb_database.json
export APPEND_KB_PATH=/path/to/agent_kb_append.json   # created empty if missing

uvicorn agent_kb_service:app --host 0.0.0.0 --port 9000
```

The service loads a `SentenceTransformer` embedding model for semantic retrieval.

## Enabling SPM in the DR path

```bash
export need_memory=true                       # turn SPM on (default: off)
export AGENT_KB_URL=http://127.0.0.1:9000      # where the service lives (this default)
```

Then run the Deep Research workflow as usual. With `need_memory` unset/false, the
DR path behaves exactly as before and never contacts the service.
