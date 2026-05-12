# SHL Assessment Recommender 🎯

A conversational AI agent that helps hiring managers find the right SHL assessments through natural dialogue. Built with FastAPI, Groq LLM, and hybrid retrieval (FAISS + keyword search).

---

## Live Demo

| Endpoint | URL |
|---|---|
| Health Check | `GET /health` |
| Chat API | `POST /chat` |
| Swagger Docs | `/docs` |

---

## Features

- **Clarifies** vague queries before recommending
- **Recommends** 1–10 SHL assessments grounded in the official catalog
- **Refines** shortlist when user changes constraints mid-conversation
- **Compares** assessments using catalog metadata only (no hallucination)
- **Refuses** off-topic requests, legal questions, and prompt-injection attempts
- **Stateless API** — full conversation history sent with every request

---

## Project Structure

```
shl-recommender/
├── main.py           # FastAPI app — /health and /chat endpoints
├── agent.py          # LLM orchestration via Groq API
├── retrieval.py      # Hybrid search (FAISS + keyword fallback)
├── catalog_data.py   # 57 SHL Individual Test Solutions catalog
├── build_index.py    # Pre-builds FAISS index (run once before server)
├── eval.py           # Evaluation harness — 10 traces, Recall@10
├── requirements.txt
├── render.yaml       # Render deployment config
├── .env.example      # Environment variable template
└── .gitignore
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| LLM | `llama-3.3-70b-versatile` via Groq | Fast (~200 tok/s), free tier, OpenAI-compatible |
| Embeddings | `all-MiniLM-L6-v2` | Lightweight, good semantic quality |
| Vector Store | FAISS IndexFlatIP | Zero infra, exact cosine similarity |
| API Framework | FastAPI + Pydantic v2 | Async, auto schema validation, Swagger UI |
| Deployment | Render / Railway | Free tier, Python native support |

---

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/shl-recommender.git
cd shl-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
Get a free key at [console.groq.com](https://console.groq.com)

Create a `.env` file:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

Or set it in PowerShell:
```powershell
$env:GROQ_API_KEY = "gsk_your_actual_key_here"
```

### 4. Build the FAISS index (run once)
```bash
python build_index.py
```

Output:
```
Loading model...
Embedding catalog...
Saved index with 57 vectors.
```

### 5. Start the server
```bash
python main.py
```

Output:
```
INFO | Loading pre-built FAISS index...
INFO | FAISS index ready — 57 vectors.
INFO | Service ready.
INFO | Uvicorn running on http://127.0.0.1:8000
```

### 6. Open in browser
```
http://127.0.0.1:8000/docs
```

---

## API Reference

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level are you looking for?"},
    {"role": "user", "content": "Mid-level, around 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java developer with stakeholder interaction.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire (OPQ32)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

**Rules:**
- `recommendations` is empty `[]` when agent is clarifying or refusing
- `recommendations` has 1–10 items when agent commits to a shortlist
- `end_of_conversation` is `true` only when task is complete
- Maximum 8 turns per conversation (user + assistant combined)
- Conversation must start with a `user` message

---

## How It Works

```
User Request
     │
     ▼
FastAPI /chat
     │
     ▼
agent.py
  ├── Off-topic / injection guard (regex fast-path, no API call)
  ├── Vague-query guard (no recommendations on first vague turn)
  ├── Retrieval: hybrid_search(conversation) → top-15 catalog items
  │       ├── FAISS semantic search (pre-built index)
  │       └── Keyword frequency boost
  ├── Context injection → augmented system prompt
  ├── Groq API (llama-3.3-70b-versatile, temp=0.3)
  └── Response validation (URL + name grounding check)
          │
          ▼
  Validated ChatResponse
```

---

## Conversation Behaviors

| Behavior | Example |
|---|---|
| **Clarify** | "I need an assessment" → asks for role and seniority |
| **Recommend** | "Java developer, mid-level" → returns 5 relevant assessments |
| **Refine** | "Add personality tests" → updates shortlist immediately |
| **Compare** | "Difference between OPQ32 and GSA?" → factual catalog-based answer |
| **Refuse** | "What are EEOC regulations?" → politely declines |

---

## Evaluation

Run the evaluation harness (with server running):

```bash
python eval.py --base-url http://localhost:8000 --verbose
```

### 10 Conversation Traces

| Trace | Tests |
|---|---|
| Java mid-level developer | Happy path clarify → recommend |
| Vague first turn | No recs on ambiguous opener |
| Senior data scientist | Python + personality tests |
| Refinement: add personality | Mid-conversation constraint update |
| OPQ32 vs GSA comparison | Grounded catalog comparison |
| EEOC legal question | Off-topic refusal |
| Prompt injection | Injection refused before API call |
| Graduate scheme | Graduate-level ability tests |
| Retail frontline | Short assessments for high volume |
| VP Engineering executive | Leadership + managerial tests |

### Scoring

| Metric | Description |
|---|---|
| Schema compliance | All fields present, correct types, ≤10 recommendations |
| Hallucination guard | All URLs and names from catalog only |
| Behavior probes | 30+ binary assertions across 10 traces |
| Recall@10 | Fraction of expected assessments in top-10 shortlist |

---

## Deployment

### Render
1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt && python build_index.py`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GROQ_API_KEY`
6. Deploy

### Railway
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Add variable: `GROQ_API_KEY`
4. Deploy

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Groq API key from console.groq.com |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model to use |
| `PORT` | No | `8000` | Server port |

---

## Catalog

57 SHL Individual Test Solutions covering:

| Type | Code | Examples |
|---|---|---|
| Ability & Aptitude | A | Verify Numerical, Verbal, Inductive Reasoning |
| Knowledge & Skills | K | Java, Python, SQL, JavaScript, DevOps |
| Personality & Behavior | P | OPQ32, OPQ32r, Motivation Questionnaire |
| Situational Judgement | B | Customer Contact, Retail, Banking Scenarios |
| Competencies/Development | C/D | UCF 360, Leadership Report |

---

## Notes

- The `.env` file is excluded from Git via `.gitignore` — never commit your API key
- `catalog.index` and `catalog_meta.pkl` are generated locally by `build_index.py`
- Cold-start delay on free hosting tiers: ~60 seconds after inactivity
- The evaluator allows up to 2 minutes for `/health` on cold-start services
