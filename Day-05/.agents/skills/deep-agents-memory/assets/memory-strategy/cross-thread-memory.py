"""
Long-term memory: cross-thread memory with CompositeBackend.

Demonstrates:
- CompositeBackend hybrid: StateBackend + StoreBackend
- Path routing: /memories/ → StoreBackend, /workspace/ → StateBackend
- Namespace tuples: (user_id, memory_type) hierarchy
- Cross-thread memory sharing
- Memory lifecycle: create, update, retrieve
- InMemoryStore (dev) vs PostgresStore (production)
"""

import asyncio
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
# from deepagents import create_deep_agent
# from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.store.memory import InMemoryStore

# --- Model Configuration ---
model = FakeListChatModel(
    responses=[
        "I've saved your preference for dark mode to long-term memory.",
        "I remember you prefer dark mode! I'll apply that to our session.",
    ]
)


# ==== Memory Backend Setup ====

# --- StateBackend: ephemeral workspace ---
# Files disappear when thread ends
# state_backend = StateBackend()

# --- StoreBackend: persistent cross-thread memory ---
# store = InMemoryStore()  # Dev; PostgresStore for production
# store_backend = StoreBackend(
#     store=store,
#     namespace=("user", "alice"),  # Base namespace
# )

# --- CompositeBackend: route by path ---
# backend = CompositeBackend(routes={
#     "/memories/":  store_backend,   # Persistent memory
#     "/workspace/": state_backend,   # Ephemeral scratch
# })


# ==== Namespace Design ====

# Namespace tuples organize cross-thread data hierarchically:
namespace_examples = {
    "User preferences": ("user", "alice", "preferences"),
    "User history": ("user", "alice", "history"),
    "Learned patterns": ("user", "alice", "patterns"),
    "Shared knowledge": ("team", "engineering", "knowledge"),
    "Project context": ("project", "webapp", "context"),
}


# ==== Memory Lifecycle Demo ====


class MemorySimulator:
    """Simulates StoreBackend memory operations for demo."""

    def __init__(self):
        self._store: dict[tuple, dict[str, dict]] = {}

    async def put(self, namespace: tuple, key: str, value: dict) -> None:
        """Create or update a memory entry."""
        if namespace not in self._store:
            self._store[namespace] = {}
        self._store[namespace][key] = value

    async def get(self, namespace: tuple, key: str) -> dict | None:
        """Retrieve a memory entry."""
        ns = self._store.get(namespace, {})
        return ns.get(key)

    async def search(self, namespace: tuple) -> list[dict]:
        """List all entries in a namespace."""
        ns = self._store.get(namespace, {})
        return [{"key": k, "value": v} for k, v in ns.items()]

    async def delete(self, namespace: tuple, key: str) -> bool:
        """Remove a memory entry."""
        ns = self._store.get(namespace, {})
        if key in ns:
            del ns[key]
            return True
        return False


# ==== Agent Setup ====

# agent = create_deep_agent(
#     model=model,
#     backend=backend,
#     checkpointer=InMemorySaver(),
#     store=store,  # Also pass store for direct access in tools
# )


# ==== Main ====


async def main():
    print("=== Cross-Thread Memory Pattern ===")
    print()

    memory = MemorySimulator()
    user_ns = ("user", "alice", "preferences")

    # 1. Create memory
    print("--- Step 1: Create Memory ---")
    await memory.put(user_ns, "theme", {"value": "dark", "confidence": 0.9})
    await memory.put(user_ns, "language", {"value": "python", "confidence": 0.8})
    print(f"  Saved: theme=dark, language=python")
    print()

    # 2. Retrieve memory (simulates different thread)
    print("--- Step 2: Retrieve from Different Thread ---")
    theme = await memory.get(user_ns, "theme")
    print(f"  Retrieved: theme={theme}")
    print(f"  (This works across threads because StoreBackend is cross-thread)")
    print()

    # 3. Update memory
    print("--- Step 3: Update Memory ---")
    await memory.put(
        user_ns, "theme", {"value": "dark", "confidence": 0.95, "source": "explicit"}
    )
    updated = await memory.get(user_ns, "theme")
    print(f"  Updated: theme={updated}")
    print()

    # 4. Search namespace
    print("--- Step 4: Search Namespace ---")
    all_prefs = await memory.search(user_ns)
    for item in all_prefs:
        print(f"  {item['key']}: {item['value']}")
    print()

    # 5. Namespace hierarchy
    print("--- Namespace Design ---")
    for label, ns in namespace_examples.items():
        print(f"  {label:25s} → {ns}")
    print()

    # 6. Path routing
    print("--- CompositeBackend Path Routing ---")
    paths = {
        "/memories/prefs.json": "StoreBackend (persistent, cross-thread)",
        "/memories/learned.json": "StoreBackend (persistent, cross-thread)",
        "/workspace/draft.md": "StateBackend (ephemeral, thread-scoped)",
        "/workspace/temp.csv": "StateBackend (ephemeral, thread-scoped)",
    }
    for path, backend in paths.items():
        print(f"  {path:35s} → {backend}")

    print()
    print("--- Store Options ---")
    print("  Dev:  InMemoryStore()           — fast, no persistence")
    print("  Prod: PostgresStore(conn_str)   — durable, cross-process")


if __name__ == "__main__":
    asyncio.run(main())
