"""
Backend patterns: CompositeBackend with longest-prefix path routing.

Demonstrates:
- CompositeBackend combining 3 backend types
- StateBackend for ephemeral workspace files
- StoreBackend for persistent cross-thread memory
- FilesystemBackend for local disk access
- Longest-prefix routing rules
"""

import asyncio
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
# from deepagents import create_deep_agent
# from deepagents.backends import (
#     CompositeBackend,
#     StateBackend,
#     StoreBackend,
#     FilesystemBackend,
# )
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.store.memory import InMemoryStore

# --- Model Configuration ---
model = FakeListChatModel(responses=[
    "I'll save the analysis to /workspace/analysis.md and remember your preference in /memories/prefs.",
])


# ==== Backend Setup ====

# --- StateBackend: ephemeral, lives in graph state ---
# Ideal for scratch work, intermediate results
# state_backend = StateBackend()

# --- StoreBackend: persistent cross-thread KV store ---
# Ideal for user preferences, learned patterns
# store = InMemoryStore()  # Dev; use PostgresStore in production
# store_backend = StoreBackend(store=store, namespace=("user", "alice"))

# --- FilesystemBackend: local disk access ---
# Ideal for project files, outputs
# fs_backend = FilesystemBackend(
#     root_dir="/path/to/project",
#     virtual_mode=False,  # True = sandbox; False = real disk
# )

# --- CompositeBackend: route by longest-prefix match ---
# composite = CompositeBackend(routes={
#     "/workspace/": state_backend,    # ephemeral scratch space
#     "/memories/":  store_backend,    # persistent memory
#     "/":           fs_backend,       # fallback to local disk
# })
# Routing: "/workspace/draft.md" → state_backend
#          "/memories/prefs"      → store_backend
#          "/src/main.py"         → fs_backend (default fallback)

# ==== Agent Setup ====

# agent = create_deep_agent(
#     model=model,
#     backend=composite,
#     checkpointer=InMemorySaver(),
# )


# ==== Main ====

async def main():
    # --- Demo: Longest-prefix routing logic ---
    print("=== CompositeBackend Routing Demo ===")
    print()

    routes = {
        "/workspace/": "StateBackend (ephemeral)",
        "/memories/":  "StoreBackend (persistent)",
        "/":           "FilesystemBackend (local disk)",
    }

    test_paths = [
        "/workspace/draft.md",
        "/workspace/analysis/results.json",
        "/memories/user_prefs",
        "/memories/learned/patterns",
        "/src/main.py",
        "/README.md",
    ]

    def resolve_backend(path: str) -> str:
        """Longest-prefix match routing (same algorithm as CompositeBackend)."""
        best_prefix = ""
        best_backend = "None"
        for prefix, backend in routes.items():
            if path.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix = prefix
                best_backend = backend
        return best_backend

    for path in test_paths:
        backend = resolve_backend(path)
        print(f"  {path:45s} → {backend}")

    print()
    print("--- BackendProtocol Methods ---")
    protocol_methods = [
        ("ls_info(path)", "List directory contents"),
        ("read(path)", "Read file content"),
        ("write(path, content)", "Write file content"),
        ("edit(path, old, new)", "Edit file with string replacement"),
        ("grep_raw(pattern, path)", "Search file contents"),
        ("glob_info(pattern)", "Find files by pattern"),
    ]
    for method, desc in protocol_methods:
        print(f"  {method:30s} — {desc}")


if __name__ == "__main__":
    asyncio.run(main())
