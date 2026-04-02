"""
Structured output example demonstrating Pydantic model response formatting.

Shows how to use response_format to get type-safe, validated responses from LLMs.

Asset: assets/02-langchain/structured-output-example/graph.py
Reference: references/02-langchain/structured-output.md
"""
import asyncio
import json
from pydantic import BaseModel, Field
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

# --- Model Configuration ---
# FakeListChatModel: testing/prototyping
# ChatOpenAI(model="gpt-4o"): direct provider
# init_chat_model("openai:gpt-4o"): universal init (recommended)

# For testing, simulate structured JSON response
structured_response = {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "555-1234",
    "company": "Acme Inc"
}
model = FakeListChatModel(responses=[json.dumps(structured_response)])

from langchain.agents import create_agent


class ContactInfo(BaseModel):
    """Contact information extracted from text."""
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    company: str = Field(default="", description="Company name if mentioned")


# Create agent with structured output
agent = create_agent(
    model=model,
    tools=[],
    response_format=ContactInfo,
)


async def extract_contact(text: str) -> ContactInfo:
    """
    Extract contact information from text using structured output.

    Args:
        text: Input text containing contact information

    Returns:
        ContactInfo object with extracted fields
    """
    result = await agent.ainvoke({
        "messages": [
            {
                "role": "system",
                "content": "You extract contact information from text. Respond with JSON matching the ContactInfo schema."
            },
            {
                "role": "user",
                "content": f"Extract contact information: {text}"
            }
        ]
    })

    # Parse structured response from the agent's last message
    last_message = result["messages"][-1]

    # Handle both direct structured response and JSON string
    if hasattr(last_message, 'structured_response'):
        return last_message.structured_response
    elif isinstance(last_message.content, str):
        # Parse JSON string to ContactInfo
        try:
            data = json.loads(last_message.content)
            return ContactInfo(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback for non-JSON responses
            print(f"Warning: Could not parse structured output, got: {last_message.content}")
            return None

    return None


async def main():
    """Demonstrate structured output extraction."""

    # Example 1: Complete contact info
    print("=== Example 1: Full Contact Information ===")
    text1 = "Contact John Doe at Acme Inc. Email: john@example.com, Phone: 555-1234"
    contact1 = await extract_contact(text1)

    if contact1:
        print(f"Name: {contact1.name}")
        print(f"Email: {contact1.email}")
        print(f"Phone: {contact1.phone}")
        print(f"Company: {contact1.company}")
        print(f"\nValidated Type: {type(contact1).__name__}")

    # Example 2: Partial info (demonstrates defaults)
    print("\n=== Example 2: Partial Information ===")
    # Update model response for partial data
    partial_response = {
        "name": "Jane Smith",
        "email": "jane.smith@tech.io",
        "phone": "555-5678",
        "company": ""  # No company mentioned
    }
    agent.model.responses = [json.dumps(partial_response)]

    text2 = "Jane Smith can be reached at jane.smith@tech.io or call 555-5678"
    contact2 = await extract_contact(text2)

    if contact2:
        print(f"Name: {contact2.name}")
        print(f"Email: {contact2.email}")
        print(f"Phone: {contact2.phone}")
        print(f"Company: {contact2.company if contact2.company else '(not provided)'}")

    # Example 3: Demonstrate validation
    print("\n=== Example 3: Pydantic Validation ===")
    try:
        # This would fail validation if email format checking was enabled
        valid_contact = ContactInfo(
            name="Test User",
            email="test@example.com",
            phone="555-0000",
            company="Test Corp"
        )
        print(f"✓ Valid contact created: {valid_contact.name}")
        print(f"  Model fields: {list(valid_contact.model_fields.keys())}")
    except ValueError as e:
        print(f"✗ Validation error: {e}")

    print("\n=== Structured Output Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
