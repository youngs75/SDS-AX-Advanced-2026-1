from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, add_messages


class AgentStateCustom(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class AgentState2Custom(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


build_1 = StateGraph(AgentState2Custom)

build_1.add_node("node1", lambda x: {"result": "node1_1"})
build_1.add_edge("__start__", "node1")

sub_graph_1 = build_1.compile()

build = StateGraph(AgentStateCustom)

build.add_node("node1", lambda x: {"result": "node1"})
build.add_node("node2", sub_graph_1)

build.add_edge("__start__", "node1")
build.add_edge("node1", "node2")
build.add_edge("node2", "__end__")

graph = build.compile()

if __name__ == "__main__":
    graph.get_graph(xray=2).draw_mermaid_png(output_file_path="./graph_2.png")
