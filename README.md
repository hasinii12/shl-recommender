# SHL Assessment Recommender

Conversational agent that helps hiring managers find the right SHL Individual Test Solutions through natural dialogue.

## Quick Start

```bash
# 1. Clone / unzip
cd shl_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 4. Run
python main.py
# → Service running at http://localhost:8000
```

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Architecture

```
User HTTP Request
       │
       ▼
  FastAPI /chat
       │
       ▼
  agent.py
  ├── Off-topic / injection guard (regex fast-path)
  ├── Vague-query guard (regex + turn count)
  ├── Retrieval: hybrid_search(conversation_history) → top-15 catalog items
  │       ├── Semantic search (FAISS + sentence-transformers all-MiniLM-L6-v2)
  │       └── Keyword boost (TF-like scoring)
  ├── Context injection → augmented system prompt
  ├── Anthropic Claude API (claude-sonnet-4-20250514, temp=0.3)
  └── Response validation (URL + name grounding check)
       │
       ▼
  Validated ChatResponse
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM | Claude Sonnet (claude-sonnet-4) | Best instruction following, JSON output reliability |
| Embeddings | all-MiniLM-L6-v2 (22M params) | Fast, lightweight, good semantic quality for short texts |
| Vector store | FAISS IndexFlatIP | Zero infra overhead, exact cosine similarity, 50-item catalog fits in RAM |
| Retrieval | Hybrid (semantic + keyword boost) | Keyword precision + semantic recall, handles abbreviations (e.g. "OPQ") |
| Response format | Strict JSON via system prompt | Deterministic parsing, evaluator-compatible |
| Temperature | 0.3 | Consistent, grounded responses; avoid hallucination |
| State | Stateless (history in request) | Matches spec, horizontally scalable |
| Grounding | Post-hoc URL+name validation | Hard safety net against hallucination independent of prompt |

## Evaluation

```bash
# Run with server already up
python eval.py --base-url http://localhost:8000 --verbose
```

### Scoring

| Category | What's checked |
|---|---|
| Schema compliance | reply/recommendations/end_of_conversation types, field presence, ≤10 recs |
| Hallucination guard | All URLs and names must exist in catalog |
| Vague query | No recommendations on ambiguous first turn |
| Clarification | Agent asks a question when context is insufficient |
| Refinement | Personality/constraint additions are honored |
| Comparison | OPQ vs GSA produces grounded factual comparison |
| Refusal | Off-topic and prompt-injection get rejected |
| Recall@10 | Fraction of expected assessments appearing in shortlist |

## Deployment

### Render (recommended free tier)
1. Push to GitHub
2. Create new Web Service on [render.com](https://render.com)
3. Connect repo, set `ANTHROPIC_API_KEY` env var
4. Deploy — uses `render.yaml` config automatically

### Docker
```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... shl-recommender
```

### Fly.io
```bash
fly launch
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
fly deploy
```

## Catalog

The catalog (`catalog_data.py`) contains 50+ SHL Individual Test Solutions scraped from the SHL product catalog, covering:

- **Ability & Aptitude (A)**: Verify series, Numerical/Verbal/Inductive/Deductive Reasoning, Mechanical, Spatial, Diagrammatic
- **Knowledge & Skills (K)**: Java, Python, SQL, JavaScript, C++, .NET, Excel, DevOps, Cybersecurity, Agile, and more
- **Personality & Behavior (P)**: OPQ32, OPQ32r, Motivation Questionnaire, Personality for Frontline
- **Situational Judgement (B)**: Customer Contact, Supervisory, Call Center, Retail, Banking scenarios
- **Competencies/Development (C/D)**: UCF 360, Leadership Report, Team Impact Report

Each item includes: name, URL, test type codes, description, job levels, duration, languages, keywords, remote testing support, and adaptive/fixed-form flag.

## File Structure

```
shl_recommender/
├── main.py          # FastAPI app, endpoint definitions, request/response schemas
├── agent.py         # LLM orchestration, system prompt, response parsing, guards
├── retrieval.py     # FAISS index, semantic search, hybrid search
├── catalog_data.py  # SHL catalog (50+ assessments) with rich metadata
├── eval.py          # Evaluation harness: 10 traces, behavior probes, Recall@10
├── requirements.txt
├── Dockerfile
├── render.yaml
└── .env.example
```
