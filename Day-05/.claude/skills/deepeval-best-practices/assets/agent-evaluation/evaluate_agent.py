"""
DeepEval Agent Evaluation Example
===================================
Demonstrates trace-based and end-to-end agent evaluation
using DeepEval's agent metrics (TaskCompletion, ToolCorrectness, etc.).
"""

import json
from deepeval import evaluate
from deepeval.metrics import (
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe


# =============================================================================
# 1. Define Agent with Tracing (for trace-only metrics)
# =============================================================================
@observe(type="tool")
def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Simulated flight search tool."""
    return [
        {"id": "FL123", "price": 450, "airline": "AirOne"},
        {"id": "FL456", "price": 380, "airline": "SkyWay"},
    ]


@observe(type="tool")
def book_flight(flight_id: str) -> dict:
    """Simulated flight booking tool."""
    return {"confirmation": "CONF-789", "flight_id": flight_id, "status": "booked"}


@observe(type="agent")
def travel_agent(user_input: str) -> str:
    """
    A simple travel agent that searches for flights and books the cheapest one.
    Replace with your actual agent implementation.
    """
    # Simulated reasoning: extract parameters from user input
    origin = "NYC"
    destination = "LA"
    date = "2025-03-15"

    # Search for flights
    flights = search_flights(origin, destination, date)

    # Find cheapest flight
    cheapest = min(flights, key=lambda x: x["price"])

    # Book the cheapest flight
    booking = book_flight(cheapest["id"])

    return (
        f"Booked flight {cheapest['id']} ({cheapest['airline']}) "
        f"for ${cheapest['price']}. Confirmation: {booking['confirmation']}"
    )


# =============================================================================
# 2. Trace-Based Evaluation (TaskCompletion, StepEfficiency, PlanQuality)
# =============================================================================
def run_trace_based_evaluation():
    """Evaluate agent using trace-only metrics that analyze full execution flow."""
    task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")

    dataset = EvaluationDataset(
        goldens=[
            Golden(input="Book the cheapest flight from NYC to LA on March 15"),
            Golden(input="Find me a flight from SF to Chicago"),
        ]
    )

    for golden in dataset.evals_iterator(metrics=[task_completion]):
        travel_agent(golden.input)


# =============================================================================
# 3. End-to-End Tool Correctness (without Tracing)
# =============================================================================
def run_tool_correctness_evaluation():
    """Evaluate tool calling accuracy using expected vs actual tools."""
    test_cases = [
        LLMTestCase(
            input="Search for flights from NYC to London",
            actual_output="Found flights: FL123 ($450), FL456 ($380)",
            tools_called=[
                ToolCall(name="search_flights"),
                ToolCall(name="book_flight"),
            ],
            expected_tools=[
                ToolCall(name="search_flights"),
            ],
        ),
        LLMTestCase(
            input="Book flight FL456",
            actual_output="Booked flight FL456. Confirmation: CONF-789",
            tools_called=[
                ToolCall(name="book_flight"),
            ],
            expected_tools=[
                ToolCall(name="book_flight"),
            ],
        ),
    ]

    tool_correctness = ToolCorrectnessMetric(threshold=0.7)
    argument_correctness = ArgumentCorrectnessMetric(threshold=0.7)

    results = evaluate(
        test_cases=test_cases,
        metrics=[tool_correctness, argument_correctness],
    )

    return results


# =============================================================================
# 4. Tool Correctness with Detailed ToolCall Parameters
# =============================================================================
def run_detailed_tool_evaluation():
    """Evaluate tool usage with detailed parameters (name, description, reasoning)."""
    test_case = LLMTestCase(
        input="Find and book the cheapest flight from NYC to Paris",
        actual_output="Booked flight FL456 for $380.",
        tools_called=[
            ToolCall(
                name="search_flights",
                description="Search available flights",
                reasoning="Need to find available flights for the route",
                input_parameters={
                    "origin": "NYC",
                    "destination": "Paris",
                    "date": "2025-03-15",
                },
                output=json.dumps([
                    {"id": "FL123", "price": 450},
                    {"id": "FL456", "price": 380},
                ]),
            ),
            ToolCall(
                name="book_flight",
                description="Book a specific flight",
                reasoning="Book the cheapest flight found",
                input_parameters={"flight_id": "FL456"},
                output=json.dumps({"confirmation": "CONF-789"}),
            ),
        ],
        expected_tools=[
            ToolCall(name="search_flights"),
            ToolCall(name="book_flight"),
        ],
    )

    metric = ToolCorrectnessMetric(
        threshold=0.7,
        should_consider_ordering=True,
    )

    metric.measure(test_case)
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")


# =============================================================================
# 5. Component-Level Agent Evaluation (with Tracing)
# =============================================================================
def run_component_level_evaluation():
    """Attach metrics directly to observed functions for component-level eval."""
    from deepeval.tracing import observe, update_current_span

    tool_correctness = ToolCorrectnessMetric()

    @observe(type="llm", metrics=[tool_correctness])
    def call_llm(messages: list[dict]) -> dict:
        """Simulated LLM call that decides which tools to use."""
        # Your actual LLM call here
        update_current_span(
            test_case=LLMTestCase(
                input=messages[0]["content"],
                actual_output="Searching for flights...",
                tools_called=[ToolCall(name="search_flights")],
                expected_tools=[ToolCall(name="search_flights")],
            )
        )
        return {"tool": "search_flights", "args": {"origin": "NYC", "destination": "LA"}}

    @observe(type="agent")
    def my_agent(user_input: str):
        messages = [{"role": "user", "content": user_input}]
        result = call_llm(messages)
        return f"Completed: {result}"

    dataset = EvaluationDataset(goldens=[Golden(input="Find flights to LA")])
    for golden in dataset.evals_iterator():
        my_agent(golden.input)


if __name__ == "__main__":
    print("=== Trace-Based Agent Evaluation ===")
    run_trace_based_evaluation()

    print("\n=== End-to-End Tool Correctness ===")
    run_tool_correctness_evaluation()

    print("\n=== Detailed Tool Evaluation ===")
    run_detailed_tool_evaluation()
