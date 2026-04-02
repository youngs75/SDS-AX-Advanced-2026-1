# Data Analysis Workflow

## Read This When

- Building data analysis workflows with sandboxed execution
- Need to analyze CSV/Excel files with agents
- Creating visualizations (charts, dashboards) programmatically
- Integrating external sharing (Slack, email) for analysis results
- Setting up secure Python execution for data work

## Skip This When

- Not doing data analysis or CSV processing with agents
- Agent doesn't need to execute Python code
- Working with non-tabular data types

## Official References

1. https://docs.langchain.com/oss/python/deepagents/data-analysis - Why: Data analysis patterns, sandbox integration, and workflow examples
2. https://docs.langchain.com/oss/python/deepagents/sandboxes - Why: Sandboxed Python execution for data work

## Core Guidance

### 1. Data Analysis Workflow

Typical pipeline:

1. Upload data files to sandbox
2. Agent analyzes data using pandas/matplotlib in sandboxed Python
3. Agent generates visualizations
4. Download results (reports, charts)
5. Optionally share via external integrations

### 2. Sandbox Setup for Data Analysis

```python
from deepagents import create_deep_agent
from deepagents.sandboxes import ModalSandbox

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    backend=ModalSandbox(
        setup_script="pip install pandas matplotlib seaborn numpy",
    ),
    system_prompt="""You are a data analyst.

Workflow:
1. Read uploaded data files
2. Analyze with pandas
3. Create visualizations with matplotlib/seaborn
4. Write summary report
5. Save outputs for download
""",
)
```

### 3. File Upload/Download Pattern

```python
import csv
from pathlib import Path
from langchain_core.messages import HumanMessage

# Upload CSV data
csv_data = Path("sales_data.csv").read_bytes()
agent.backend.upload_files([("/data/sales.csv", csv_data)])

# Run analysis
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Analyze sales trends and create a dashboard")]},
    config={"configurable": {"thread_id": "analysis_1"}},
)

# Download generated charts
outputs = agent.backend.download_files([
    "/output/dashboard.png",
    "/output/report.md",
])
```

### 4. Sandboxed Python Execution

Agent uses `execute` tool to run Python in the sandbox:

```python
# Agent generates and executes code like:
execute("""
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/data/sales.csv')
monthly = df.groupby('month')['revenue'].sum()
monthly.plot(kind='bar', title='Monthly Revenue')
plt.savefig('/output/revenue_chart.png', dpi=150, bbox_inches='tight')
print(f'Processed {len(df)} rows, {len(monthly)} months')
"
""")
```

### 5. External Integration

Custom tools for sharing results:

```python
from langchain.tools import tool

@tool
async def send_slack_message(channel: str, message: str, file_path: str | None = None) -> str:
    """Send analysis results to a Slack channel."""
    # Slack API integration
    return f"Sent to #{channel}"

@tool
async def send_email_report(to: str, subject: str, body: str, attachments: list[str] | None = None) -> str:
    """Email analysis report with attachments."""
    # Email API integration
    return f"Sent to {to}"

agent = create_deep_agent(
    model=model,
    tools=[send_slack_message, send_email_report],
    backend=sandbox,
)
```

### 6. LangSmith Tracing

Track analysis workflow:

```python
import os
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "data-analysis"
# All agent steps, tool calls, and sandbox executions are traced
```

### 7. Best Practices

| Practice | Why |
|----------|-----|
| Use sandbox for all code execution | Isolation prevents local system damage |
| Set `interrupt_on={"execute": True}` for untrusted data | Review generated code before running |
| Use `write_todos` for multi-step analysis | Track analysis progress |
| Save intermediate results to files | Recoverable if analysis fails mid-way |
| Download outputs explicitly | Sandbox may be ephemeral |

## Quick Checklist

- [ ] Is a sandbox configured with required Python packages?
- [ ] Are data files uploaded before analysis begins?
- [ ] Is execute tool gated with interrupt_on for untrusted data?
- [ ] Are outputs downloaded before sandbox teardown?
- [ ] Is LangSmith tracing enabled for workflow observability?
- [ ] Are external integrations (Slack, email) using @tool with proper schemas?

## Next File

- Return to router: `../00-layer-model.md`
