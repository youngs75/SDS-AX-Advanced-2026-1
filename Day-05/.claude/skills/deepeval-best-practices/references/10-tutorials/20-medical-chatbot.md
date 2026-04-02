# Tutorial: Medical Chatbot Evaluation

## Read This When
- Want to learn by building a multi-turn conversational medical chatbot with RAG, tools (appointment booking), and chat memory using LangChain + Qdrant
- Need a complete example of ConversationalTestCase, user simulation (ConversationSimulator), and multi-turn metrics (TurnRelevancyMetric, TurnFaithfulnessMetric, RoleAdherenceMetric)
- Looking for an end-to-end tutorial covering development, evaluation, improvement (model/prompt iteration), and production tracing with `@observe`

## Skip This When
- Building a single-turn RAG QA system rather than a multi-turn chatbot -- see `references/10-tutorials/30-rag-qa-agent.md`
- Need the guide-level procedure for agent evaluation without a full tutorial -- see `references/09-guides/20-agent-evaluation.md`
- Want API reference for conversational metrics -- see `references/03-eval-metrics/30-agent-metrics.md`

---

## Overview

This tutorial teaches how to develop and evaluate a reliable LLM-powered medical chatbot using OpenAI, LangChain, Qdrant, and DeepEval — covering the full lifecycle from initial development through production deployment.

**Technologies**: OpenAI, LangChain, Qdrant (vector DB), SentenceTransformers, DeepEval

**What is evaluated**:
- Symptom diagnosis accuracy (RAG faithfulness)
- Response relevancy across multi-turn conversations
- Role adherence and knowledge retention
- Safety compliance

---

## Stage 1: Development

### Architecture

The chatbot is a multi-turn conversational agent with:
- A RAG pipeline over a medical knowledge base (Gale Encyclopedia of Alternative Medicine)
- An appointment booking tool
- LangChain chat memory (session-scoped)

### Section 1: Setup Your Model

Create a `MedicalChatbot` class using LangChain's chat models:

```python
from langchain_openai import ChatOpenAI

class MedicalChatbot:
    def __init__(self, model: str):
        self.model = ChatOpenAI(model=model)
        # Choose the LLM that will drive the agent
        # Only certain models support this so ensure your model supports it as well
```

Test invocation:

```python
chatbot = MedicalChatbot(model="gpt-4o-mini")
chatbot.model.invoke([{"user": "Hi!"}])
```

Expected output:

```
AIMessage(
    content="Hey, how can I help you today?",
    additional_kwargs={},
    response_metadata={
        'prompt_feedback': {'block_reason': 0, 'safety_ratings': []},
        'finish_reason': 'STOP',
        'model_name': 'gpt-4o-mini',
        'safety_ratings': []
    },
    id='run--c2786aa1-75c4-4644-ae59-9327a2e8c153-0',
    usage_metadata={'input_tokens': 23, 'output_tokens': 417, 'total_tokens': 440, 'input_token_details': {'cache_read': 0}}
)
```

> **Note:** Alternative interfaces to OpenAI or other models are acceptable.

### Section 2: Create RAG Pipeline For Diagnosis

The tutorial uses "The Gale Encyclopedia of Alternative Medicine" (text version) as the medical knowledge base. Users must download and convert to `.txt` format locally.

#### Index Medical Knowledge

```python
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

class MedicalChatbot:
    def __init__(self, model: str):
        self.model = ChatOpenAI(model=model)
        # For RAG engine
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")

    def index_knowledge(self, document_path: str):
        with open(document_path) as file:
            documents = file.readlines()
        # Create namespace in qdrant
        self.client.create_collection(
            collection_name="gale_encyclopedia",
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            ),
        )
        # Vectorize and index into qdrant
        self.client.upload_points(
            collection_name="gale_encyclopedia",
            points=[
                models.PointStruct(
                    id=idx,
                    vector=self.encoder.encode(doc).tolist(),
                    payload={"content": doc}
                ) for idx, doc in enumerate(documents)
            ],
        )
```

Initialize knowledge base:

```python
chatbot = MedicalChatbot()
chatbot.index_knowledge("path-to-your-encyclopedia.txt")
```

> **Note:** Execute `index_knowledge` only once.

#### Query Knowledge Base (RAG Retrieval Tool)

```python
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

class MedicalChatbot:
    def __init__(self, model: str):
        self.model = ChatOpenAI(model=model)
        # For RAG engine
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")

    @tool
    def retrieve_knowledge(self, query: str) -> str:
        """"A tool to retrive data on various diagnosis methods from gale encyclopedia"""
        hits = self.client.query_points(
            collection_name="gale_encyclopedia",
            query=self.encoder.encode(query).tolist(),
            limit=3
        ).points
        contexts = [hit.payload['content'] for hit in hits]
        return "\n".join(contexts)

    def index_knowledge(self, document_path: str):
        # Same as above
        pass
```

Test query:

```python
chatbot = MedicalChatbot()
chatbot.retrieve_knowledge("Cough, fever, and diarrhea.")
```

> **Info:** The `@tool` decorator enables LangChain to invoke the method as a function call.

### Section 3: Create Tool To Book Appointments

#### Appointment Data Model

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date

class Appointment(BaseModel):
    id: str
    name: str
    email: str
    date: date
    symptoms: Optional[List[str]] = Field(default=None)
    diagnosis: Optional[str] = Field(default=None)
```

#### Implement create_appointment Tool

```python
import uuid

class MedicalChatbot:
    def __init__(self, model: str):
        self.model = ChatOpenAI(model=model)
        # For RAG engine
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        # For managing appointments
        self.appointments: List[Appointment] = []

    @tool
    def create_appointment(self, name: str, email: str, date: str) -> str:
        """Create a new appointment with the given ID, name, email, and date"""
        try:
            appointment = Appointment(
                id=str(uuid.uuid4()),
                name=name,
                email=email,
                date=date.fromisoformat(date)
            )
            self.appointments.append(appointment)
            return f"Created new appointment with ID: {appointment.id} for {name} on {date}."
        except ValueError:
            return f"Invalid date format. Please use YYYY-MM-DD format."

    @tool
    def retrieve_knowledge(self, query: str) -> str:
        # Same as above
        pass

    def index_knowledge(self, document_path: str):
        # Same as above
        pass
```

### Section 4: Implementing Chat Histories

#### Session History Helper

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Simple in-memory store for chat histories
chat_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]
```

#### Complete Agent Setup

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import StructuredTool

class MedicalChatbot:
    def __init__(self, model: str, system_prompt: str):
        self.model = ChatOpenAI(model=model)
        self.system_prompt = system_prompt
        # For RAG engine
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        # For managing appointments
        self.appointments: List[Appointment] = []
        # Setup agent with memory
        self.setup_agent()

    def setup_agent(self):
        """Setup the agent with tools and memory"""
        # Create prompt messages
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        # Create agent
        tools = [
            StructuredTool.from_function(func=self.retrieve_knowledge),
            StructuredTool.from_function(func=self.create_appointment)
        ]
        agent = create_tool_calling_agent(self.model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        self.agent_with_memory = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    # Other methods from above goes here
    ...
```

### Section 5: Eyeball Your First Output

#### Interactive Session Implementation

```python
from typing import Optional

def start_session(session_id: Optional[str] = None):
    """Start an interactive session with the chatbot"""
    print("Hello! I am Baymax, your personal healthcare companion.")
    print("How are you feeling today? (type 'exit' to quit.)")
    while True:
        if session_id is None:
            session_id = str(uuid.uuid4())
        user_input = input("Your query: ")
        if user_input.lower() == 'exit':
            break
        response = chatbot.agent_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print("Baymax:", response["output"])

# These parameters will be evaluated later
MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """"""

# Initialize chatbot and start session
chatbot = MedicalChatbot(model=MODEL, system_prompt=SYSTEM_PROMPT)
chatbot.index_knowledge("path-to-your-encyclopedia.txt")
start_session()
```

Execute application:

```bash
python main.py
```

Example conversation:

```
Hello! I am Baymax, your personal healthcare companion.
How are you feeling today? (type 'exit' to quit.")
Your query: Hello Baymax, I've been feeling unwell. I have a fever and a sore throat.
Baymax: I'm sorry to hear you're not feeling well. A fever and sore throat are common
symptoms of upper respiratory infections. Based on medical knowledge, rest, fluids, and
over-the-counter pain relievers can help. If your symptoms are severe or persistent,
it's best to consult a doctor. Would you like me to help you schedule an appointment?

Your query: Yes, please book an appointment for tomorrow at 2 PM.
Baymax: I'll help you create an appointment. I need your name and email to proceed.
Could you please provide those details?

Your query: My name is John Doe and my email is john@example.com
Baymax: Created new appointment with ID: 550e8400-e29b-41d4-a716-446655440000 for
John Doe on 2024-01-16.
```

> **Tip:** Manual evaluation ("eyeballing") is not scalable or reliable, especially for extended conversations. Model and system prompt are the optimization variables for subsequent sections.

---

## Stage 2: Evaluation

### Core Concept: ConversationalTestCase Structure

Multi-turn interactions are represented as `ConversationalTestCase` objects composed of `Turn` elements:

```python
from deepeval.test_case import ConversationalTestCase, Turn

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="I've a sore throat."),
        Turn(role="assistant", content="Thanks for letting me know?"),
    ]
)
```

**Key Principle**: Evaluations should run on consistent scenario sets across chatbot iterations to establish valid benchmarks and detect regressions.

### Three Approaches to Creating Test Cases

#### Option 1: Use Historical Conversations (Quick but backward-looking)

```python
from deepeval.test_case import ConversationalTestCase, Turn

# Example: Fetch conversations from your database
conversations = fetch_conversations_from_db()  # Your database query here
test_cases = []

for conv in conversations:
    turns = [Turn(role=msg["role"], content=msg["content"])
             for msg in conv["messages"]]
    test_case = ConversationalTestCase(turns=turns)
    test_cases.append(test_case)

print(test_cases)
```

**Advantages**: Quick execution; data already exists

**Limitations**: "Ad-hoc insights into past performance" only; cannot reliably predict new version performance; backward-looking results

#### Option 2: Manual Prompting (Forward-looking but not scalable)

```python
from deepeval.test_case import ConversationalTestCase, Turn

# Initialize test case list
test_cases = []

def start_session(chatbot: MedicalChatbot):
    turns = []
    while True:
        user_input = input("Your query: ")
        if user_input.lower() == 'exit':
            break

        # Call chatbot
        response = chatbot.agent_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        # Add turns to list
        turns.append(Turn(role="user", content=user_input))
        turns.append(Turn(role="assistant", content=response["output"]))
        print("Baymax:", response["output"])

# Initialize chatbot and start session
chatbot = MedicalChatbot(model="...", system_prompt="...")
start_session(chatbot)

# Print test cases
print(test_cases)
```

**Advantages**: Tests current system version; produces forward-looking insights

**Limitations**: Extremely time-consuming; not scalable

#### Option 3: User Simulations (Recommended)

Automatically simulate conversations based on standardized scenarios, avoiding manual interaction while testing current system versions.

**Step 1: Create Goldens Dataset**

```python
from deepeval.dataset import EvaluationDataset, ConversationalGolden

goldens = [
    ConversationalGolden(
        scenario="User with a sore throat asking for paracetamol.",
        expected_outcome="Gets a recommendation for panadol."
    ),
    ConversationalGolden(
        scenario="Frustrated user looking to rebook their appointment.",
        expected_outcome="Gets redirected to a human agent"
    ),
    ConversationalGolden(
        scenario="User just looking to talk to somebody.",
        expected_outcome="Tell them this chatbot isn't meant for this use case."
    )
]

# Create dataset and optionally push to Confident AI
dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="Medical Chatbot Dataset")
```

> **Note**: Minimum 20 goldens recommended for adequate dataset size; each golden produces a single test case.

**Step 2: Simulate Conversations**

```python
from deepeval.test_case import Turn
from deepeval.simulator import ConversationSimulator

# Wrap your chatbot in a callback function
def model_callback(input, turns: List[Turn], thread_id: str) -> Turn:
    # 1. Get latest simulated user input
    user_input = turns[-1].content

    # 2. Call chatbot
    response = chatbot.agent_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    # 3. Return chatbot turn
    return Turn(role="assistant", content=response["output"])

simulator = ConversationSimulator(model_callback=model_callback)
test_cases = simulator.simulate(goldens=dataset.goldens)
```

**Benefits**:
- Tests against current system version
- Fully automated process
- Creates consistent benchmarks across iterations
- Enables straightforward performance comparisons

### Metrics for Multi-Turn Evaluation

#### TurnRelevancyMetric

Generic metric applicable to virtually any use case. Evaluates each assistant turn using a sliding window approach to construct unit interactions with historical context.

```python
from deepeval.metrics import TurnRelevancyMetric

relevancy = TurnRelevancyMetric()
```

**Use Case**: Most common metric; extremely generic and useful as evaluation criteria for both single and multi-turn use cases.

#### TurnFaithfulnessMetric

Domain-specific metric assessing whether chatbot responses contradict retrieval context. Essential when the system uses external knowledge sources (e.g., medical encyclopedias for diagnosis).

```python
from deepeval.metrics import TurnFaithfulnessMetric

faithfulness = TurnFaithfulnessMetric()
```

**Purpose**: Assessing whether there are any contradictions between the retrieval context in a turn to the generated assistant content.

### Running the First Multi-Turn Evaluation

```python
from deepeval import evaluate

# Test cases and metrics from previous sections
evaluate(
    test_cases=[test_cases],
    metrics=[relevancy, faithfulness],
    hyperparameters={
        "Model": MODEL,  # The model used in your agent
        "Prompt": SYSTEM_PROMPT  # The system prompt used in your agent
    }
)
```

Results are stored in Confident AI. Run `deepeval view` to open the dashboard. Each relevancy and faithfulness score ties to the specific model and prompt version, enabling easy comparison when updating either parameter.

---

## Stage 3: Improvement

### Pulling Datasets

Retrieve previously created datasets from the cloud:

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="Medical Chatbot Dataset")
```

Using conversational goldens to generate test cases:

```python
from deepeval.simulator import ConversationSimulator
from typing import List, Dict
from medical_chatbot import MedicalChatbot
import asyncio

medical_chatbot = MedicalChatbot()

async def model_callback(input: str, conversation_history: List[Dict[str, str]]) -> str:
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(None, medical_chatbot.agent_executer.invoke, {
        "input": input,
        "chat_history": conversation_history
    })
    return res["output"]

for golden in dataset.goldens:
    simulator = ConversationSimulator(
        user_intentions=golden.additional_metadata["user_intentions"],
        user_profiles=golden.additional_metadata["user_profiles"]
    )
    convo_test_cases = simulator.simulate(
        model_callback=model_callback,
        stopping_criteria="Stop when the user's medical concern is addressed with actionable advice.",
    )
    for test_case in convo_test_cases:
        test_case.scenario = golden.scenario
        test_case.expected_outcome = golden.expected_outcome
        test_case.chatbot_role = "a professional, empathetic medical assistant"
    print(f"\nGenerated {len(convo_test_cases)} conversational test cases.")
```

### Enhanced System Prompt

```
You are BayMax, a friendly and professional healthcare chatbot. You assist users by
retrieving accurate information from the Gale Encyclopedia of Medicine and helping them
book medical appointments.

Your key responsibilities:
- Provide clear, fact-based health information from trusted sources only.
- Retrieve and summarize relevant entries from the Gale Encyclopedia when asked.
- Help users schedule or manage healthcare appointments as needed.
- Maintain a warm, empathetic, and calm tone.
- Always recommend consulting a licensed healthcare provider for diagnoses or treatment.

Do not:
- Offer medical diagnoses or personal treatment plans.
- Speculate or give advice beyond verified sources.
- Ask for sensitive personal information unless necessary for booking.

Use phrases like:
- "According to the Gale Encyclopedia of Medicine..."
- "This is general information. Please consult a healthcare provider for advice."

Your goal is to support users with reliable, respectful healthcare guidance.
```

### Iterating Over Models and Prompts

```python
from deepeval.metrics import (
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
    ConversationalGEval,
)
from deepeval.dataset import EvaluationDataset, ConversationalGolden
from deepeval.simulator import ConversationSimulator
from typing import List, Dict
from deepeval import evaluate
from medical_chatbot import MedicalChatbot

dataset = EvaluationDataset()
dataset.pull(alias="Medical Chatbot Dataset")

metrics = [knowledge_retention, role_adherence, safety_check]
models = ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"]
system_prompt = "..."  # Use your new system prompt here

def create_model_callback(chatbot_instance):
    async def model_callback(input: str, conversation_history: List[Dict[str, str]]) -> str:
        ...
    return model_callback

for model in models:
    for golden in dataset.goldens:
        simulator = ConversationSimulator(
            user_intentions=golden.additional_metadata["user_intentions"],
            user_profiles=golden.additional_metadata["user_profiles"]
        )
        chatbot = MedicalChatbot("gale_encyclopedia.txt", model)
        chatbot.setup_agent(system_prompt)
        convo_test_cases = simulator.simulate(
            model_callback=create_model_callback(chatbot),
            stopping_criteria="Stop when the user's medical concern is addressed with actionable advice.",
        )
        for test_case in convo_test_cases:
            test_case.scenario = golden.scenario
            test_case.expected_outcome = golden.expected_outcome
            test_case.chatbot_role = "a professional, empathetic medical assistant"
        evaluate(convo_test_cases, metrics)
```

### Sample Evaluation Results

| Metric | Score |
|--------|-------|
| Knowledge Retention | 0.8 |
| Role Adherence | 0.7 |
| Safety Check | 0.9 |

GPT-4 achieved the best performance across all three metrics.

### Updated MedicalChatbot with Configurable Parameters

```python
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from deepeval.tracing import observe

class MedicalChatbot:
    def __init__(
        self,
        document_path,
        model="gpt-4",
        encoder="all-MiniLM-L6-v2",
        memory=":memory:",
        system_prompt=""
    ):
        self.model = ChatOpenAI(model=model)
        self.appointments = {}
        self.encoder = SentenceTransformer(encoder)
        self.client = QdrantClient(memory)
        self.store_data(document_path)
        self.system_prompt = system_prompt or (
            "You are a virtual health assistant designed to support users with symptom understanding and appointment management. Start every conversation by actively listening to the user's concerns. Ask clear follow-up questions to gather information like symptom duration, intensity, and relevant health history. Use available tools to fetch diagnostic information or manage medical appointments. Never assume a diagnosis unless there's enough detail, and always recommend professional medical consultation when appropriate."
        )
        self.setup_agent(self.system_prompt)

    def store_data(self, document_path):
        ...

    @tool
    @observe()
    def query_engine(self, query: str) -> str:
        ...

    @tool
    def create_appointment(self, appointment_id: str) -> str:
        ...

    def setup_tools(self):
        ...

    @observe()
    def setup_agent(self, system_prompt: str):
        ...

    @observe()
    def interactive_session(self, session_id):
        ...
```

Usage with all configurable parameters:

```python
from medical_chatbot import MedicalChatbot

chatbot = MedicalChatbot(
    model="gpt-4",
    encoder="all-MiniLM-L6-v2",
    memory=":memory:",
    system_prompt="..."
)
```

---

## Stage 4: Production (Evals in Prod)

### Setup Tracing with @observe

Apply `@observe` decorators throughout the chatbot to enable component-level evaluation and debugging visibility. Metrics are evaluated at the span level during live operation.

```python
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.metrics import ContextualRelevancyMetric

class MedicalChatbot:
    def __init__(
        self,
        document_path,
        model="gpt-4",
        encoder="all-MiniLM-L6-v2",
        memory=":memory:",
        system_prompt=""
    ):
        self.model = ChatOpenAI(model=model)
        self.appointments = {}
        self.encoder = SentenceTransformer(encoder)
        self.client = QdrantClient(memory)
        self.store_data(document_path)
        self.system_prompt = system_prompt or (
            "You are a virtual health assistant designed to support users with "
            "symptom understanding and appointment management. Start every "
            "conversation by actively listening to the user's concerns. Ask clear "
            "follow-up questions to gather information like symptom duration, "
            "intensity, and relevant health history. Use available tools to fetch "
            "diagnostic information or manage medical appointments. Never assume a "
            "diagnosis unless there's enough detail, and always recommend "
            "professional medical consultation when appropriate."
        )
        self.setup_agent(self.system_prompt)

    def store_data(self, document_path):
        ...

    @tool
    @observe(metrics=[ContextualRelevancyMetric()], type="retriever")
    def query_engine(self, query: str) -> str:
        """A tool to retrieve data on various diagnosis methods from gale encyclopedia"""
        hits = self.client.search(
            collection_name="gale_encyclopedia",
            query_vector=self.encoder.encode(query).tolist(),
            limit=3,
        )
        contexts = [hit.payload['content'] for hit in hits]
        # Update the Retriever span
        update_current_span(
            input=query,
            retrieval_context=contexts
        )
        return "\n".join(contexts)

    @observe(type="agent")
    def interactive_session(self, session_id):
        print("Hello! I am Baymax, your personal health care companion.")
        print("Please enter your symptoms or ask about appointment details. Type 'exit' to quit.")
        while True:
            user_input = input("Your query: ")
            if user_input.lower() == 'exit':
                break
            response = self.agent_with_chat_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            update_current_trace(
                thread_id=session_id,
                input=user_input,
                output=response["output"]
            )
            print("Agent Response:", response["output"])
```

> **Note:** Adding `@observe` decorators to all functions helps evaluate your entire workflow without interrupting application flow.

### Evaluating Production Traces

Once tracing is configured, evaluate the full conversation thread using metric collections defined in Confident AI:

```python
from deepeval.tracing import evaluate_thread

evaluate_thread(thread_id="your-thread-id", metric_collection="Metric Collection")
```

Metric collections can be created on the Confident AI platform for online evaluations to detect regressions and bugs.

### Key Production Principles

- Add `@observe` decorators to all functions in the chatbot pipeline
- Use `update_current_span()` to provide test case data to each span
- Use `update_current_trace()` to link conversation turns to a `thread_id`
- Use `evaluate_thread()` to trigger async evaluation after conversation ends
- Metric collections on Confident AI are reusable across builds
