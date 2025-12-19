# 🛡️ AI-Powered Incident Management System

> **An intelligent incident response platform using RAG (Retrieval Augmented Generation) and AI agents**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)

---

## 📚 Table of Contents

1. [What is This Project?](#what-is-this-project)
2. [Core Concepts Explained](#core-concepts-explained)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Understanding the Code](#understanding-the-code)
6. [Tech Stack](#tech-stack)
7. [API Reference](#api-reference)
8. [Future Improvements](#future-improvements)

---

## 🤔 What is This Project?

### The Problem
When a production system goes down (like a website crashing or payments failing), engineers need to:
1. **Understand what's happening** - What's broken? How bad is it?
2. **Find the cause** - Was it a recent code deployment? Database issue?
3. **Fix it quickly** - What steps should we take? Have we seen this before?

### The Solution
This project is an **AI-powered assistant** that helps engineers during incidents by:
- **Finding similar past incidents** - "We had this exact problem 3 months ago, here's how we fixed it"
- **Identifying suspicious changes** - "A deployment happened 2 hours before this incident started"
- **Recommending playbooks** - "Here's a step-by-step guide for this type of issue"
- **Answering questions** - Chat interface to ask questions about the incident

### What Makes It Special?
It uses **RAG (Retrieval Augmented Generation)** - a technique where AI finds relevant information first, then generates responses based on that context. This makes AI responses much more accurate and relevant.

---

## 🧠 Core Concepts Explained

### 1. RAG (Retrieval Augmented Generation)

**What it is:** Instead of asking an AI to answer from memory, you first search a database for relevant information, then give that info to the AI along with the question.

**Why it matters:** 
- AI responses are grounded in real data
- Reduces hallucinations (AI making stuff up)
- Can use your company's private data

**How it works in this project:**
```python
User asks: "What caused this incident?"
     ↓
Step 1: Search vector database for:
  - Similar past incidents
  - Recent deployments
  - Relevant playbooks
     ↓
Step 2: Combine all found context
     ↓
Step 3: Generate response using context
```

### 2. Vector Embeddings & Similarity Search

**What it is:** Converting text into numbers (vectors) so we can mathematically compare how similar two pieces of text are.

**Simple Example:**
```python
"Payment failed" → [0.8, 0.2, 0.5, ...]  (128 numbers)
"Transaction error" → [0.75, 0.25, 0.48, ...]  (128 numbers)
"Database slow" → [0.1, 0.9, 0.3, ...]  (128 numbers)

Similarity("Payment failed", "Transaction error") = 0.95 (very similar!)
Similarity("Payment failed", "Database slow") = 0.30 (not similar)
```

**In this project:** `ai/vector_store.py` handles all of this

### 3. AI Agent

**What it is:** A system that can:
- Understand user requests
- Decide what actions to take
- Use tools (like search) to gather information
- Generate helpful responses

**In this project:** `ai/agent.py` - The agent analyzes incidents and answers questions

### 4. Evaluation Framework

**What it is:** A way to measure how good the AI's responses are.

**Metrics we track:**
- **Relevance** - Does the response match the incident context?
- **Accuracy** - Are the facts correct?
- **Helpfulness** - Is it actually useful?
- **Latency** - How fast is the response?

**In this project:** `ai/evaluation.py` handles all metrics

---

## 📁 Project Structure

```
incident-management-ai/
│
├── app.py                    # MAIN UI - Streamlit dashboard (START HERE)
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
│
├── api/                      # REST API (optional, for programmatic access)
│   ├── __init__.py
│   └── main.py              # FastAPI server with all endpoints
│
├── ai/                       # AI CORE - The brain of the system
│   ├── __init__.py
│   ├── vector_store.py      # Vector database for semantic search
│   ├── agent.py             # AI agent that analyzes incidents
│   └── evaluation.py        # Metrics and evaluation framework
│
├── models/                   # DATA MODELS - Defines data structures
│   ├── __init__.py
│   └── schemas.py           # Pydantic models (Incident, Playbook, etc.)
│
├── data/                     # SAMPLE DATA
│   ├── __init__.py
│   └── seed_data.py         # Sample incidents, playbooks, deployments
│
└── venv/                     # Python virtual environment (created on setup)
```

### File-by-File Explanation

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `app.py` | Main Streamlit UI | `main()`, `render_incident_detail()`, `render_ai_chat()` |
| `ai/vector_store.py` | Semantic search | `VectorStore`, `search()`, `find_similar_incidents()` |
| `ai/agent.py` | AI logic | `IncidentAIAgent`, `analyze_incident()`, `answer_question()` |
| `ai/evaluation.py` | Metrics | `AIEvaluationFramework`, `record_response()`, `get_metrics()` |
| `models/schemas.py` | Data types | `Incident`, `Playbook`, `AIInsight`, `Severity` |
| `data/seed_data.py` | Sample data | `INCIDENTS`, `PLAYBOOKS`, `DEPLOYMENTS` |
| `api/main.py` | REST API | `/api/ai/analyze`, `/api/ai/chat`, `/api/metrics` |

---

## 🚀 How to Run

### First Time Setup

```bash
# Clone the repository
git clone https://github.com/harmehak0173/incident-management-ai.git
cd incident-management-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### Option 1: Streamlit UI (Recommended)
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run Streamlit
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

#### Option 2: FastAPI Backend Only
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run FastAPI server
python -m uvicorn api.main:app --reload --port 8000
```
API docs at http://localhost:8000/docs

### Stopping the Application
Press `Ctrl+C` in the terminal to stop.

---

## 🔍 Understanding the Code

### How an Incident Analysis Works

```python
# 1. User selects an incident in the UI
incident = Incident(
    title="Payment Processing Failures",
    severity="critical",
    services=["Payment Service", "API Gateway"]
)

# 2. Agent retrieves context using RAG
context = {
    "similar_incidents": [...]    # Past incidents that look similar
    "recent_deployments": [...]   # Code changes before incident
    "relevant_playbooks": [...]   # Step-by-step guides
}

# 3. Agent generates insights
insights = [
    AIInsight(
        type="similar_pattern",
        title="Similar Incident Found",
        content="This matches an incident from Jan 15...",
        confidence=0.87
    ),
    AIInsight(
        type="root_cause", 
        title="Recent Deployment Detected",
        content="payment-service v2.3.4 was deployed 2 hours ago...",
        confidence=0.75
    )
]

# 4. Response shown to user with suggested actions
```

### Vector Search Flow

```python
# In ai/vector_store.py

# 1. When data loads, we create embeddings for everything
doc = Document(
    id="incident-001",
    content="Payment Processing Failures - Stripe Integration...",
    embedding=[0.8, 0.2, 0.5, ...]  # Computed from content
)

# 2. When user searches, we compare embeddings
query = "payment errors"
query_embedding = compute_embedding(query)  # [0.75, 0.22, 0.48, ...]

# 3. Find most similar documents using cosine similarity
for doc in documents:
    similarity = cosine_similarity(query_embedding, doc.embedding)
    if similarity > 0.5:  # Threshold
        results.append(doc)
```

### Evaluation Flow

```python
# In ai/evaluation.py

# 1. Every AI response is automatically evaluated
metrics = auto_evaluate(response, context)
# Returns: {relevance: 0.85, accuracy: 0.78, helpfulness: 0.90}

# 2. User can also give feedback (thumbs up/down)
record_feedback(evaluation_id, helpful=True)

# 3. Metrics dashboard shows trends over time
```

---

## 🛠️ Tech Stack

| Technology | What It Does | Why We Use It |
|------------|--------------|---------------|
| **Python 3.11+** | Programming language | Industry standard for AI/ML |
| **FastAPI** | Backend web framework | Fast, modern, automatic API docs |
| **Streamlit** | UI framework | Quick beautiful dashboards |
| **Pydantic** | Data validation | Type-safe data models |
| **NumPy** | Numerical computing | Vector math for embeddings |
| **Uvicorn** | ASGI server | Runs FastAPI |

### What Could Be Added for Production

| Current (Demo) | Production Alternative |
|----------------|----------------------|
| NumPy vectors | Pinecone, Weaviate, ChromaDB |
| No real LLM | OpenAI GPT-4, Anthropic Claude |
| In-memory data | PostgreSQL, MongoDB |
| No auth | OAuth, JWT tokens |

---

## 📡 API Reference

### Endpoints (when running FastAPI)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/incidents` | List all incidents |
| GET | `/api/incidents/{id}` | Get specific incident |
| POST | `/api/ai/analyze` | Analyze an incident |
| POST | `/api/ai/chat` | Chat about an incident |
| POST | `/api/ai/search` | Search knowledge base |
| GET | `/api/metrics` | Get AI performance metrics |

### Example API Call

```bash
# Analyze an incident
curl -X POST http://localhost:8000/api/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{"incident_id": "inc-001"}'
```

---

## 🔮 Future Improvements

### Easy Additions
- [ ] Connect to real OpenAI/Claude API
- [ ] Add more sample data
- [ ] Dark mode toggle

### Medium Complexity
- [ ] Use ChromaDB for persistent vector storage
- [ ] Add user authentication
- [ ] Real-time updates via WebSocket

### Advanced
- [ ] Fine-tune embedding model on incident data
- [ ] Automated runbook execution
- [ ] Slack/Teams integration
- [ ] Multi-tenant support

---

## 🎯 Key Takeaways for Learning

1. **RAG Pattern**: Search first, then generate. This is the foundation of modern AI applications.

2. **Vector Embeddings**: Text → Numbers → Similarity comparison. This enables semantic search.

3. **Evaluation Matters**: Always measure AI performance. Without metrics, you're flying blind.

4. **Clean Architecture**: Separate concerns (AI logic, data models, UI, API) makes code maintainable.

5. **Type Safety**: Pydantic models catch bugs early and make code self-documenting.

---

## 📝 Notes

- This is a **demonstration prototype** showcasing AI/ML engineering capabilities
- The "AI" currently uses heuristics, not actual LLM calls
- Sample data is fake but realistic
- Vector embeddings are simplified for demo purposes
- Ready for integration with real LLMs (OpenAI, Claude, etc.)

---

## 📄 License

MIT License - feel free to use this project for learning or as a foundation for your own incident management system.

---

## 👨‍💻 Author

Built by **Harmehak Singh Khangura** to demonstrate expertise in:
- RAG-based AI systems
- Vector embeddings and semantic search
- AI agent development
- Production-ready Python engineering
- FastAPI and Streamlit

---
