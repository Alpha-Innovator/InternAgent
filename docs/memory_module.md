## Memory Module

The memory module is a new feature in InternAgent 1.5. It maintains a persistent record of past experiments across sessions — each completed idea is stored with its metrics and a label (positive / neutral / negative relative to baseline). During idea generation, the agent queries this history to avoid repeating approaches that have already failed and to build on directions that have worked.

The module has two independent components:

- **Retrieval** — queries the accumulated records at idea generation time and injects a guidance summary into the generation prompt
- **Saving** — writes a record for each idea immediately after its experiments complete

The two are controlled by separate flags because they are useful in different combinations:

| Saving | Retrieval | When to use |
|--------|-----------|-------------|
| on | off | First session — nothing to retrieve yet, but start building the store |
| on | on | Subsequent sessions — use history and continue adding to it |
| off | on | Read-only evaluation — query history without adding new records |
| off | off | Disable the module entirely |

**When is it useful?**

The memory module provides no benefit on a first run, since there is nothing in the store yet. Its value grows with repeated use on the same task: after several sessions, the generation agent can avoid rediscovering the same dead ends and has a concrete basis for preferring directions with a track record of success.

---

### Configuration

#### Retrieval

Controlled by `memory.task_memory` in the config file.

```yaml
memory:
  task_memory:
    enabled: false          # Set to true to enable retrieval during idea generation
    memory_dir: "./config/mem_store"  # Where the memory store is read from
    top_k: 5                # Number of similar past records to surface per query
    alpha: 0.5              # Search balance: 1.0 = keyword only, 0.0 = semantic only
                            # Tune toward 1.0 when the store is sparse (few records)
    embedding_mode: "description"  # Which part of the idea to embed: title / description / method / full
    embedding:
      model_type: "local"   # local / openai / azure
      model_name: ""        # Path to local model directory (see below), or API model name
```

Retrieval requires a working embedding model for the semantic search component. If `model_name` is empty or the model fails to load, the module is silently skipped.

#### Saving

Controlled by `memory.online_memory` in the config file.

```yaml
memory:
  online_memory:
    enabled: false          # Set to true to auto-save results after each experiment
    aggregation: "best"     # How to summarize multi-run metrics into one record:
                            #   best — use the best run (default)
                            #   avg  — average across all runs
                            #   last — use the final run
```

Saving also requires a working embedding model (the record is embedded immediately on write).

---

### Setting up the embedding model

Both components require a local sentence embedding model. Download one before enabling either flag.

If HuggingFace is reachable:
```bash
huggingface-cli download BAAI/bge-base-en-v1.5 --local-dir ./models/bge-base-en-v1.5
```

If not (e.g. on a restricted network), try the mirror:
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-base-en-v1.5 --local-dir ./models/bge-base-en-v1.5
```

Or via ModelScope:
```bash
pip install modelscope
modelscope download --model BAAI/bge-base-en-v1.5 --local_dir ./models/bge-base-en-v1.5
```

Then set `model_name` to the download path:
```yaml
memory:
  task_memory:
    embedding:
      model_name: "./models/bge-base-en-v1.5"
```

The memory store is written to `memory_dir` (default: `./config/mem_store`) and persists across sessions. Records from previous runs are automatically loaded on the next run.

---

### Long Memory (IdeaGraph and PromptEvolver)

In addition to task memory and online memory, InternAgent 1.5 includes a long memory component that tracks the full history of ideas and automatically evolves the generation prompt over time.

It has two sub-components:

- **IdeaGraph** — builds a graph of all ideas generated across sessions, connecting similar ones by embedding similarity. Used to detect redundancy and measure exploration breadth.
- **PromptEvolver** — periodically rewrites the generation prompt based on accumulated experience, nudging the agent toward directions that have worked and away from those that haven't.

```yaml
memory:
  long_memory:
    enabled: true
    idea_graph:
      similarity_threshold: 0.7   # Minimum similarity to draw an edge between two ideas;
                                   # lower = more connections, higher = only near-duplicates linked
    prompt_evolver:
      enabled: true               # Set to false to disable automatic prompt evolution
      evolution_interval: 1       # Evolve the prompt every N loop rounds
                                  #   1 = every round (aggressive adaptation)
                                  #   2 = every other round (more conservative)
```

Long memory requires the same embedding model as task memory. If no embedding model is configured, it falls back to keyword-only idea tracking without semantic edges.


---

## Strategy-Procedural Memory (SPM) — Deep Research path

> **Note:** the memory described above (task / online / long memory) belongs to the
> **algorithm-discovery / idea-generation** path. The **Deep Research** path
> (`DRAgent` / `internagent/mas/agents/dr_agents`) uses a separate mechanism:
> **Strategy-Procedural Memory (SPM)**, described in Section 2.4.1 of the paper.
> The two are independent subsystems.

SPM is an experience knowledge base that accumulates lessons from completed research
tasks and reuses them on similar future subtasks. Each task distills into two kinds of
experience:

| SPM component | stored field | prompt label |
|---|---|---|
| **Strategy memory** (planning) | `agent_experience` | `PLANNING EXPERIENCE` |
| **Procedural memory** (execution) | `search_agent_experience` | `EXECUTION EXPERIENCE` |

**Where it lives (`internagent/mas/agents/dr_agents`):**

- **Planner-side retrieval** — `agents/global_planner_agent.py`: before building the task graph it retrieves aligned planning experience from the KB and injects it via `GLOBAL_PLANNER_MEMORY_BLOCK` (appended to `GLOBAL_PLANNER_PROMPT`), guiding the planner toward more coherent reasoning graphs. An external `set_memory_prompt(...)` hook is also available.
- **Executor-side retrieval** — `agents/task/execution_agent.py`: before running a subtask it calls
  `call_hybrid_search('base'/'append', task)` and injects the results into
  `EXECUTION_PROMPT_WITHMEMORY` (defined in `prompts/default_prompts.py`).
- **Write-back** — `workflow/task.py`: after a task completes it distills the trace with
  `MEMORY_REASONING_PROMPT` (into strategy + procedural experience) and calls
  `call_appendkb(...)`.
- **Store / retrieval service** — `agent_kb/` (a standalone FastAPI service with a
  base + append two-tier KB and hybrid text+semantic retrieval). See
  [`internagent/mas/agents/dr_agents/agent_kb/README.md`](../internagent/mas/agents/dr_agents/agent_kb/README.md).

### Enabling SPM

SPM is **off by default** and only activates when both hold:

1. `need_memory=true` — env var (recommended; overrides config), or the `need_memory`
   keys under `global_execution` / `global_execution.execution` in the DR `config_*.yaml`.
2. The `agent_kb` service is running and reachable at `AGENT_KB_URL`
   (default `http://127.0.0.1:9000`).

```bash
# 1) start the experience KB service (needs KB data + an embedding model)
cd internagent/mas/agents/dr_agents/agent_kb
pip install -r requirements.txt
export BASE_KB_PATH=/path/to/agent_kb_database.json
export APPEND_KB_PATH=/path/to/agent_kb_append.json
uvicorn agent_kb_service:app --host 0.0.0.0 --port 9000

# 2) enable SPM for the Deep Research run
export need_memory=true
export AGENT_KB_URL=http://127.0.0.1:9000
```

If the package or service is unavailable, the DR path **degrades gracefully**: retrieval
is skipped (the normal `EXECUTION_PROMPT` is used) and write-back is a no-op — the task
still completes. The large KB data files are not bundled; configure them via the env
vars above.


## 📝 Citation

```bibtex
@article{feng2026internagent,
  title={InternAgent-1.5: A Unified Agentic Framework for Long-Horizon Autonomous Scientific Discovery},
  author={Shiyang Feng and Runmin Ma and Xiangchao Yan and Yue Fan and Yusong Hu and Songtao Huang and Shuaiyu Zhang and Zongsheng Cao and Tianshuo Peng and Jiakang Yuan and Zijie Guo and Zhijie Zhong and Shangheng Du and Weida Wang and Jinxin Shi and Yuhao Zhou and Xiaohan He and Zhiyin Yu and Fangchen Yu and Bihao Zhan and Qihao Zheng and Jiamin Wu and Mianxin Liu and Chi Zhang and Shaowei Hou and Shuya Li and Yankai Jiang and Wenjie Lou and Lilong Wang and Zifu Wang and Jiong Wang and Wanghan Xu and Yue Deng and Dongrui Liu and Yiheng Wang and Wenlong Zhang and Fenghua Ling and Shufei Zhang and Xiaosong Wang and Shuangjia Zheng and Xun Huang and Siqi Sun and Shuyue Hu and Peng Ye and Chunfeng Song and Bin Wang and Conghui He and Yihao Liu and Xin Li and Qibin Hou and Tao Chen and Xiangyu Yue and Bin Wang and Liang He and Dahua Lin and Bowen Zhou and Bo Zhang and Lei Bai},
  journal={arXiv preprint arXiv:2602.08990},
  year={2026}
}
```

