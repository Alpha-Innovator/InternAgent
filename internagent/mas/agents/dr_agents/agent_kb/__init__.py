"""
agent_kb — Strategy-Procedural Memory (SPM) experience knowledge base for the
Deep Research path.

This package implements the SPM described in the paper (Section 2.4.1):
- Strategy memory   -> planning experience   (``agent_experience``)
- Procedural memory -> execution experience  (``search_agent_experience``)

Layout:
- ``example.py``          : lightweight HTTP client used by the DR agents at runtime
                            (only depends on ``requests``).
- ``memory_utils.py``     : helpers to strip tool messages and parse distilled experience.
- ``agent_kb_service.py`` : standalone FastAPI service (base + append KB, hybrid retrieval).
- ``agent_kb_retrieval.py``/``agent_kb_utils.py``/``kb_prompt.py`` : service internals.

The service and its KB data are optional and configured out-of-band (see README.md).
The DR agents import the client lazily and degrade gracefully when the service is
unavailable, so importing this package must never be a hard requirement.
"""
