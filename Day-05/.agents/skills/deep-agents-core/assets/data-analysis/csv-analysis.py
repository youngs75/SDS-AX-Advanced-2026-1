"""
Data analysis: CSV analysis workflow with sandboxed execution.

Demonstrates:
- CSV analysis pipeline: upload → analyze → visualize → share
- Sandbox Python execution (pandas, matplotlib, seaborn)
- File transfer: upload_files() / download_files()
- Custom integration tools (@tool for Slack/email)
- LangSmith tracing configuration
"""

import asyncio
from langchain_core.language_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
# from deepagents import create_deep_agent
# from deepagents.sandbox import ModalSandbox  # or DaytonaSandbox, RunloopSandbox
# from langgraph.checkpoint.memory import InMemorySaver

# --- Model Configuration ---
model = FakeListChatModel(responses=[
    "I'll analyze the sales data CSV. Let me upload it to the sandbox first.",
    "Analysis complete! Revenue is up 23% YoY. I've created a 6-panel dashboard.",
])


# ==== Custom Integration Tools ====

@tool
async def send_slack_message(channel: str, message: str) -> str:
    """Send analysis results to a Slack channel."""
    # Production: Slack API call
    return f"Sent to #{channel}: {message[:50]}..."

@tool
async def send_email_report(to: str, subject: str, body: str) -> str:
    """Send analysis report via email."""
    # Production: email API call
    return f"Email sent to {to}: {subject}"


# ==== Analysis Pipeline ====

pipeline_steps = [
    {
        "step": "1. Upload",
        "description": "Transfer CSV to sandbox via upload_files()",
        "code": 'await sandbox.upload_files({"data.csv": csv_bytes})',
    },
    {
        "step": "2. Analyze",
        "description": "Run pandas analysis in sandboxed Python",
        "code": """await sandbox.execute('''
import pandas as pd
df = pd.read_csv('data.csv')
summary = df.describe()
print(summary.to_json())
''')""",
    },
    {
        "step": "3. Visualize",
        "description": "Generate charts with matplotlib/seaborn",
        "code": """await sandbox.execute('''
import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# ... chart code ...
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
''')""",
    },
    {
        "step": "4. Download",
        "description": "Retrieve results from sandbox",
        "code": 'files = await sandbox.download_files(["dashboard.png", "report.md"])',
    },
    {
        "step": "5. Share",
        "description": "Send results via custom tools",
        "code": 'await send_slack_message("#analytics", summary)',
    },
]


# ==== Sandbox Configuration ====

sandbox_options = {
    "ModalSandbox": {
        "best_for": "GPU-accelerated ML, large datasets",
        "cold_start": "~2-5s",
        "packages": "Pre-installed: pandas, numpy, scikit-learn, matplotlib",
    },
    "DaytonaSandbox": {
        "best_for": "Quick analyses, fast cold start",
        "cold_start": "~1-2s",
        "packages": "Install on demand via pip",
    },
    "RunloopSandbox": {
        "best_for": "One-off explorations, disposable environments",
        "cold_start": "~3-5s",
        "packages": "Full development environment",
    },
}


# ==== Agent Setup ====

# Analysis agent with sandbox and sharing tools
# sandbox = ModalSandbox(gpu="T4")  # or DaytonaSandbox(), RunloopSandbox()
# agent = create_deep_agent(
#     model=model,  # or "anthropic:claude-sonnet-4-20250514"
#     tools=[send_slack_message, send_email_report],
#     sandbox=sandbox,
#     checkpointer=InMemorySaver(),
#     system_prompt=(
#         "You are a data analyst. When given CSV data:\n"
#         "1. Upload to sandbox\n"
#         "2. Run pandas analysis\n"
#         "3. Create visualizations\n"
#         "4. Share results via Slack\n"
#     ),
# )


# ==== Main ====

async def main():
    print("=== CSV Analysis Workflow ===")
    print()

    # Pipeline steps
    print("--- Analysis Pipeline ---")
    for step_info in pipeline_steps:
        print(f"  {step_info['step']}: {step_info['description']}")
        print(f"    Code: {step_info['code'].split(chr(10))[0]}...")
        print()

    # Sandbox options
    print("--- Sandbox Options ---")
    for name, info in sandbox_options.items():
        print(f"  {name}:")
        for key, value in info.items():
            print(f"    {key:15s}: {value}")
        print()

    # Demo tool execution
    print("--- Integration Demo ---")
    result = await send_slack_message.ainvoke({
        "channel": "analytics",
        "message": "Revenue up 23% YoY. Dashboard attached.",
    })
    print(f"  Slack: {result}")

    result = await send_email_report.ainvoke({
        "to": "team@example.com",
        "subject": "Q4 Sales Analysis",
        "body": "Revenue increased 23% year-over-year...",
    })
    print(f"  Email: {result}")
    print()

    # LangSmith tracing
    print("--- LangSmith Tracing ---")
    print("  Set environment variables for tracing:")
    print('    LANGSMITH_API_KEY="ls_..."')
    print('    LANGSMITH_PROJECT="csv-analysis"')
    print('    LANGSMITH_TRACING="true"')
    print("  Traces show: tool calls, sandbox execution, analysis steps")
    print()

    # Best practices
    print("--- Best Practices ---")
    print("  1. Upload only necessary columns (reduce data transfer)")
    print("  2. Set pandas display options for concise output")
    print("  3. Save figures at 150 DPI (quality vs size balance)")
    print("  4. Use sandbox.execute() for ALL data operations (security)")
    print("  5. Never pass raw data through agent context (token waste)")


if __name__ == "__main__":
    asyncio.run(main())
