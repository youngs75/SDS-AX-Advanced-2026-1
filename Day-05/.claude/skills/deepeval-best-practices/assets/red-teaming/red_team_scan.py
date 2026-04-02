"""
DeepEval Red Teaming Example
==============================
Demonstrates automated red teaming for LLM safety testing using DeepTeam,
including vulnerability scanning, attack enhancements, and risk assessment.

Prerequisites:
    pip install deepteam
"""

from deepteam import red_team
from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import (
    Bias,
    Toxicity,
    PIILeakage,
    PromptLeakage,
    ExcessiveAgency,
)
from deepteam.attacks.single_turn import (
    PromptInjection,
    Roleplay,
    Base64,
    ROT13,
    Leetspeak,
    GrayBox,
)
from deepteam.attacks.multi_turn import (
    LinearJailbreaking,
    CrescendoJailbreaking,
)
from deepteam.test_case import RTTurn


# =============================================================================
# 1. Define the Model Callback
# =============================================================================
async def model_callback(input: str, turns: list[RTTurn] = None) -> str:
    """
    Wraps your target LLM system. Replace the body with your actual LLM call.

    Parameters:
        input: The adversarial attack string
        turns: Conversation history for multi-turn attacks (None for single-turn)

    Returns:
        RTTurn with the assistant's response.
        Can also return a plain string for simpler setups.
    """
    # Replace with your actual LLM application logic
    # Example: response = await your_llm.generate(input, history=turns)
    response_text = f"I'm a helpful assistant. Regarding your query: {input}"

    return RTTurn(
        role="assistant",
        content=response_text,
        # Optional: include retrieval context for RAG evaluation
        # retrieval_context=["Your retrieval context here"],
        # Optional: include tool calls for agentic evaluation
        # tools_called=[ToolCall(name="SearchDatabase")],
    )


# =============================================================================
# 2. Quick Start with red_team() Function
# =============================================================================
def run_quick_scan():
    """
    Simplest way to run red teaming with the red_team() function.
    Tests a small set of vulnerabilities with single-turn attacks.
    """
    # Define vulnerabilities to test
    bias = Bias(types=["race", "gender"])
    toxicity = Toxicity(types=["insults", "profanity"])

    # Define attack methods
    prompt_injection = PromptInjection()
    base64_attack = Base64()

    # Run the red team scan
    risk_assessment = red_team(
        model_callback=model_callback,
        vulnerabilities=[bias, toxicity],
        attacks=[prompt_injection, base64_attack],
        attacks_per_vulnerability_type=3,
        target_purpose="A helpful customer support chatbot for an e-commerce platform.",
    )

    # View results
    print(risk_assessment)

    # Save results locally
    risk_assessment.save(to="./deepteam-results/")

    return risk_assessment


# =============================================================================
# 3. Comprehensive Scan with RedTeamer Class
# =============================================================================
def run_comprehensive_scan():
    """
    Use RedTeamer for better lifecycle control, attack reuse,
    and more configuration options.
    """
    red_teamer = RedTeamer(
        target_purpose="Provide financial advice, investment suggestions, "
        "and answer user queries related to personal finance.",
        simulator_model="gpt-3.5-turbo-0125",
        evaluation_model="gpt-4o",
        async_mode=True,
        max_concurrent=10,
    )

    # Configure diverse vulnerabilities
    vulnerabilities = [
        Bias(types=["race", "gender", "religion"]),
        Toxicity(types=["insults", "threats"]),
        PIILeakage(types=["direct_disclosure", "api_db_access", "session_leak"]),
        PromptLeakage(types=["secrets_and_credentials", "instructions"]),
        ExcessiveAgency(types=["functionality", "permissions", "autonomy"]),
    ]

    # Mix single-turn and multi-turn attacks with weights
    attacks = [
        # Encoding-based (fastest, no LLM calls)
        Base64(weight=1),
        ROT13(weight=1),
        Leetspeak(weight=1),
        # One-shot (one LLM call each)
        PromptInjection(weight=2),
        Roleplay(weight=1),
        GrayBox(weight=1),
    ]

    # Run the scan
    risk_assessment = red_teamer.red_team(
        model_callback=model_callback,
        vulnerabilities=vulnerabilities,
        attacks=attacks,
        attacks_per_vulnerability_type=5,
        ignore_errors=False,
    )

    # View aggregated vulnerability scores
    print("Vulnerability Scores:")
    print(red_teamer.vulnerability_scores)

    # View detailed per-attack breakdown
    print("\nDetailed Breakdown:")
    print(red_teamer.vulnerability_scores_breakdown)

    return red_teamer, risk_assessment


# =============================================================================
# 4. Multi-Turn Attack Scan (for Chatbots)
# =============================================================================
def run_multi_turn_scan():
    """
    Use multi-turn attacks for chatbot and conversational applications.
    Multi-turn attacks adapt across conversation turns for deeper probing.
    """
    red_teamer = RedTeamer(
        target_purpose="An AI-powered chatbot that helps users manage their accounts.",
        simulator_model="gpt-3.5-turbo-0125",
        evaluation_model="gpt-4o",
        async_mode=True,
        max_concurrent=5,
    )

    # Multi-turn attacks iterate and adapt based on model responses
    multi_turn_attacks = [
        LinearJailbreaking(),
        CrescendoJailbreaking(),
    ]

    # Combine with single-turn for comprehensive coverage
    single_turn_attacks = [
        PromptInjection(weight=2),
        Base64(weight=1),
    ]

    risk_assessment = red_teamer.red_team(
        model_callback=model_callback,
        vulnerabilities=[
            PIILeakage(types=["direct_disclosure", "social_manipulation"]),
            Bias(types=["race"]),
        ],
        attacks=single_turn_attacks + multi_turn_attacks,
        attacks_per_vulnerability_type=3,
    )

    print("Multi-turn scan results:")
    print(risk_assessment)

    return risk_assessment


# =============================================================================
# 5. Interpreting Risk Assessment Results
# =============================================================================
def interpret_results(red_teamer):
    """
    Analyze and interpret red teaming results.
    Scores are binary: 0 (vulnerable) or 1 (secure).
    """
    # Average scores per vulnerability
    print("=== Vulnerability Scores ===")
    print(red_teamer.vulnerability_scores)

    # Detailed per-attack breakdown as DataFrame
    breakdown = red_teamer.vulnerability_scores_breakdown

    # Filter for specific vulnerability issues
    if "Excessive Agency" in breakdown["Vulnerability"].values:
        excessive_agency_issues = breakdown[
            breakdown["Vulnerability"] == "Excessive Agency"
        ]
        print("\n=== Excessive Agency Issues ===")
        print(excessive_agency_issues)

    # Scores near 1.0 = strong performance (secure)
    # Scores near 0.0 = potential vulnerabilities (needs attention)
    print("\n=== Summary ===")
    print("Scores near 1.0 indicate the model is robust against that vulnerability.")
    print("Scores near 0.0 indicate the model may be vulnerable and needs improvement.")


if __name__ == "__main__":
    print("=== Quick Red Team Scan ===")
    run_quick_scan()

    print("\n=== Comprehensive Scan with RedTeamer ===")
    red_teamer, assessment = run_comprehensive_scan()

    print("\n=== Interpreting Results ===")
    interpret_results(red_teamer)
