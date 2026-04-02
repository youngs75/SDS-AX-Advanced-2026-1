"""
DeepEval Conversation and Multi-Turn Evaluation Example
==========================================================
Demonstrates multi-turn chatbot evaluation with turn-level and
conversation-level metrics using ConversationalTestCase.
"""

from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase, TurnParams
from deepeval.metrics import (
    TurnRelevancyMetric,
    TurnFaithfulnessMetric,
    TurnContextualRelevancyMetric,
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    ConversationalGEval,
)


# =============================================================================
# 1. Build a ConversationalTestCase from Chat History
# =============================================================================
def build_test_case_from_history():
    """
    Construct a ConversationalTestCase from a typical chatbot conversation.
    Each Turn has a role, content, and optional retrieval_context.
    Replace the simulated data with your actual chat logs.
    """
    convo_test_case = ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="What is your return policy?",
            ),
            Turn(
                role="assistant",
                content="We offer a 30-day full refund at no extra cost. "
                "Simply contact our support team to initiate the return.",
                retrieval_context=[
                    "All customers are eligible for a 30 day full refund at no extra cost.",
                    "Returns can be initiated by contacting our support team.",
                ],
            ),
            Turn(
                role="user",
                content="How long does the refund take to process?",
            ),
            Turn(
                role="assistant",
                content="Refunds are typically processed within 5-7 business days "
                "after we receive the returned item.",
                retrieval_context=[
                    "Refunds are processed within 5-7 business days after item receipt.",
                ],
            ),
            Turn(
                role="user",
                content="Do you offer free shipping on exchanges?",
            ),
            Turn(
                role="assistant",
                content="Yes, all exchanges come with free return shipping. "
                "We will also cover the shipping for the replacement item.",
                retrieval_context=[
                    "Exchanges include free return and replacement shipping.",
                    "Premium members receive priority processing on exchanges.",
                ],
            ),
        ],
    )

    return convo_test_case


# =============================================================================
# 2. Turn-Level Metrics (Per-Turn RAG Quality)
# =============================================================================
def evaluate_turn_level():
    """
    Evaluate individual turns within a conversation.
    Turn-level metrics average scores across all assistant turns.
    """
    convo_test_case = build_test_case_from_history()

    # TurnRelevancyMetric: Are assistant responses relevant to the conversation?
    # Does NOT require retrieval_context
    turn_relevancy = TurnRelevancyMetric(
        threshold=0.5,
        window_size=10,
    )

    # TurnFaithfulnessMetric: Are responses grounded in retrieval_context?
    # REQUIRES retrieval_context in each assistant Turn
    turn_faithfulness = TurnFaithfulnessMetric(
        threshold=0.5,
        window_size=10,
    )

    # TurnContextualRelevancyMetric: Is the retrieval_context relevant to the query?
    # REQUIRES retrieval_context in each assistant Turn
    turn_contextual_relevancy = TurnContextualRelevancyMetric(
        threshold=0.5,
        window_size=10,
    )

    # Run evaluation with all turn-level metrics
    results = evaluate(
        test_cases=[convo_test_case],
        metrics=[turn_relevancy, turn_faithfulness, turn_contextual_relevancy],
    )

    return results


# =============================================================================
# 3. Conversation-Level Metrics (Whole Conversation Quality)
# =============================================================================
def evaluate_conversation_level():
    """
    Evaluate the entire conversation as a whole.
    These metrics assess overall chatbot performance across all turns.
    """
    convo_test_case = ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="I need to return these shoes and also check if you have my size in blue.",
            ),
            Turn(
                role="assistant",
                content="I'd be happy to help with both! For the return, we offer a 30-day refund. "
                "Let me also check our inventory for blue shoes in your size.",
            ),
            Turn(
                role="user",
                content="Great. My size is 10. Also, what are your store hours?",
            ),
            Turn(
                role="assistant",
                content="We have blue running shoes in size 10 available! "
                "Our store hours are Monday through Saturday, 9 AM to 8 PM.",
            ),
            Turn(
                role="user",
                content="Perfect, and what size did I say I was?",
            ),
            Turn(
                role="assistant",
                content="You mentioned you wear size 10.",
            ),
        ],
    )

    # ConversationCompletenessMetric: Did the chatbot satisfy all user intentions?
    completeness = ConversationCompletenessMetric(threshold=0.5)

    # KnowledgeRetentionMetric: Does the chatbot remember facts from earlier turns?
    knowledge_retention = KnowledgeRetentionMetric(threshold=0.5)

    results = evaluate(
        test_cases=[convo_test_case],
        metrics=[completeness, knowledge_retention],
    )

    return results


# =============================================================================
# 4. ConversationalGEval for Custom Criteria
# =============================================================================
def evaluate_custom_criteria():
    """
    Define custom evaluation criteria for conversations using ConversationalGEval.
    This is the optimal method for evaluating custom chatbot standards.
    """
    convo_test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="Hello, I need help with a refund."),
            Turn(
                role="assistant",
                content="Of course! I'd be happy to help you with the refund process. "
                "Could you please provide your order number?",
            ),
            Turn(role="user", content="It's ORDER-12345."),
            Turn(
                role="assistant",
                content="Thank you! I've found your order. I'll initiate the refund now. "
                "You should receive it within 5-7 business days.",
            ),
        ],
    )

    # Define a professionalism metric
    professionalism = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant maintained a professional and "
        "courteous tone throughout the conversation based on the content.",
        evaluation_params=[TurnParams.CONTENT],
        threshold=0.7,
    )

    # Define a helpfulness metric
    helpfulness = ConversationalGEval(
        name="Helpfulness",
        criteria="Evaluate whether the assistant proactively addressed the user's "
        "needs and provided clear, actionable guidance based on the content.",
        evaluation_params=[TurnParams.CONTENT],
        threshold=0.7,
    )

    # Define a metric that considers retrieval context
    grounded_accuracy = ConversationalGEval(
        name="Grounded Accuracy",
        criteria="Assess whether the assistant's responses are factually accurate "
        "and grounded in the provided retrieval context.",
        evaluation_params=[TurnParams.CONTENT, TurnParams.RETRIEVAL_CONTEXT],
        threshold=0.6,
    )

    results = evaluate(
        test_cases=[convo_test_case],
        metrics=[professionalism, helpfulness],
    )

    # Access individual metric scores
    for metric in [professionalism, helpfulness]:
        print(f"{metric.name}: score={metric.score}, reason={metric.reason}")

    return results


# =============================================================================
# 5. Combining Turn-Level and Conversation-Level Metrics
# =============================================================================
def evaluate_comprehensive():
    """
    Run a comprehensive evaluation combining both turn-level and
    conversation-level metrics on the same conversation.
    """
    convo_test_case = ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content="What if these shoes don't fit?",
            ),
            Turn(
                role="assistant",
                content="We offer a 30-day full refund at no extra cost.",
                retrieval_context=[
                    "All customers are eligible for a 30 day full refund at no extra cost.",
                ],
            ),
            Turn(
                role="user",
                content="How long does the refund take?",
            ),
            Turn(
                role="assistant",
                content="Refunds are processed within 5-7 business days.",
                retrieval_context=[
                    "Refunds are processed within 5-7 business days after approval.",
                ],
            ),
            Turn(
                role="user",
                content="My name is Sarah. Can you check my order status?",
            ),
            Turn(
                role="assistant",
                content="Hi Sarah! I'd be happy to help. Could you provide your order number?",
            ),
            Turn(
                role="user",
                content="It's ORDER-789. Also, do you remember my name?",
            ),
            Turn(
                role="assistant",
                content="Of course, Sarah! Let me look up ORDER-789 for you right away.",
            ),
        ],
    )

    # Turn-level metrics
    turn_relevancy = TurnRelevancyMetric(threshold=0.5)
    turn_faithfulness = TurnFaithfulnessMetric(threshold=0.5)

    # Conversation-level metrics
    completeness = ConversationCompletenessMetric(threshold=0.5)
    knowledge_retention = KnowledgeRetentionMetric(threshold=0.5)

    # Custom criteria
    tone = ConversationalGEval(
        name="Friendly Tone",
        criteria="Evaluate whether the assistant maintained a warm, friendly, "
        "and approachable tone throughout the conversation.",
        evaluation_params=[TurnParams.CONTENT],
        threshold=0.7,
    )

    results = evaluate(
        test_cases=[convo_test_case],
        metrics=[
            turn_relevancy,
            turn_faithfulness,
            completeness,
            knowledge_retention,
            tone,
        ],
    )

    return results


# =============================================================================
# 6. Building Test Cases from Real Chat Logs
# =============================================================================
def build_from_chat_log():
    """
    Helper pattern for converting raw chat log data into ConversationalTestCases.
    Replace the sample data with your actual chat log format.
    """
    # Sample chat log — replace with your actual chat data
    chat_log = [
        {"role": "user", "message": "Hi, I need help."},
        {"role": "assistant", "message": "Hello! How can I assist you today?", "sources": []},
        {"role": "user", "message": "What is your refund policy?"},
        {
            "role": "assistant",
            "message": "We offer a 30-day full refund.",
            "sources": ["30 day full refund at no extra cost."],
        },
    ]

    # Convert chat log to Turns
    turns = []
    for entry in chat_log:
        turn_kwargs = {
            "role": entry["role"],
            "content": entry["message"],
        }
        # Add retrieval_context only for assistant turns that have sources
        if entry["role"] == "assistant" and entry.get("sources"):
            turn_kwargs["retrieval_context"] = entry["sources"]

        turns.append(Turn(**turn_kwargs))

    convo_test_case = ConversationalTestCase(turns=turns)

    print(f"Built test case with {len(turns)} turns")
    return convo_test_case


if __name__ == "__main__":
    print("=== Turn-Level Evaluation ===")
    evaluate_turn_level()

    print("\n=== Conversation-Level Evaluation ===")
    evaluate_conversation_level()

    print("\n=== Custom Criteria Evaluation ===")
    evaluate_custom_criteria()

    print("\n=== Comprehensive Evaluation ===")
    evaluate_comprehensive()

    print("\n=== Build from Chat Log ===")
    build_from_chat_log()
