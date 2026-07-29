
def remove_tool_messages(messages):
    """删除 messages 中所有 role 为 'tool' 的消息，返回新列表。"""
    if not isinstance(messages, list):
        return messages
    return [m for m in messages if not (isinstance(m, dict) and m.get("role") == "tool")]

import re
from typing import Dict, Any

def extract_agent_and_search_experience(text: str) -> Dict[str, Any]:
    """
    Parse the model output in the required bullet format and extract:
    - agent_experience block
    - search_agent_experience block

    Expected structure (no JSON):
    Question: ...
    agent_experience:
      ...
    search_agent_experience:
      ...
    """

    def _block_between(s: str, start_label: str, end_label: str | None) -> str:
        # Match a label line exactly (allow trailing spaces)
        start_pat = re.compile(rf"^{re.escape(start_label)}\s*$", re.MULTILINE)
        m = start_pat.search(s)
        if not m:
            return ""

        start = m.end()

        if end_label is None:
            end = len(s)
        else:
            end_pat = re.compile(rf"^{re.escape(end_label)}\s*$", re.MULTILINE)
            m2 = end_pat.search(s, start)
            end = m2.start() if m2 else len(s)

        return s[start:end].strip()

    question_match = re.search(r"^Question:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    question = question_match.group(1).strip() if question_match else ""

    agent_experience = _block_between(
        text,
        start_label="agent_experience:",
        end_label="search_agent_experience:",
    )

    search_agent_experience = _block_between(
        text,
        start_label="search_agent_experience:",
        end_label=None,
    )

    return {
        "question": question,
        "agent_experience": agent_experience,
        "search_agent_experience": search_agent_experience,
    }


# ---------------- Example ----------------
if __name__ == "__main__":
    sample_output = """
Question: Debug tool message ordering error

agent_experience:
- Success reasons:
  - ...
- Reusable planning playbook:
  - ...
- Planning pitfalls & prevention:
  - Pitfall: ...
    Prevention: ...

search_agent_experience:
- Success reasons:
  - ...
- Reusable execution playbook:
  - ...
- Failure analysis:
  - Failure #1:
    What failed: ...
    Error: ...
    Root cause: ...
    Fix: ...
    Prevention: ...
    Early detection: ...
"""

    parsed = extract_agent_and_search_experience(sample_output)
    print("QUESTION:", parsed["question"])
    print("\nAGENT EXPERIENCE:\n", parsed["agent_experience"])
    print("\nSEARCH AGENT EXPERIENCE:\n", parsed["search_agent_experience"])

