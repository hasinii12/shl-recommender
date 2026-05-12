# Approach Document — SHL Assessment Recommender
**AI Intern Take-home Assignment | SHL Labs**

---

## 1. Problem Decomposition

The core challenge is bridging the gap between a hiring manager's vague intent ("I'm hiring a developer") and a grounded, catalog-constrained shortlist. This requires four distinct capabilities to coexist robustly: **clarification** (know when you don't know enough), **recommendation** (retrieve and rank from catalog only), **refinement** (update in-place, not restart), and **comparison** (answer factual questions from data, not model priors).

I decomposed the system into four layers with clean separation of concerns:

1. **Catalog Layer** (`catalog_data.py`) — Static ground truth. 50+ SHL Individual Test Solutions with rich metadata: test_type codes, job levels, duration, languages, keywords, remote/adaptive flags. Manually curated from the public SHL product catalog page. Every URL is canonical.

2. **Retrieval Layer** (`retrieval.py`) — Hybrid search over the catalog. Semantic similarity via FAISS + `all-MiniLM-L6-v2` embeddings (cosine similarity on normalized vectors) combined with keyword-frequency boosting. Items matching both semantic and keyword signals are ranked higher. Fallback to pure keyword scoring if embedding libraries are unavailable.

3. **Agent Layer** (`agent.py`) — Orchestrates LLM calls. Builds a rich system prompt containing the full catalog listing + a dynamically retrieved context block (top-15 relevant items per conversation). Uses Claude Sonnet at temperature=0.3 for consistency. Includes fast-path regex guards for off-topic queries and prompt-injection attempts *before* incurring API cost. Post-hoc validation strips any hallucinated names or URLs from the model's output by cross-referencing against the catalog.

4. **API Layer** (`main.py`) — FastAPI with strict Pydantic schemas matching the evaluator's spec exactly. Stateless: full conversation history in every request. Validates message ordering, enforces 8-turn cap, handles errors gracefully.

---

## 2. Retrieval & Context Engineering

**Why hybrid search?** Pure semantic search struggles with abbreviations ("OPQ", "GSA", "SJT") and exact product names. Pure keyword search misses semantic relationships ("data scientist" → Python, SQL, numerical reasoning). Combining both gives recall on known terms and precision on novel phrasings.

**Retrieval-augmented prompting:** Each `/chat` call extracts all user text from the conversation history, runs hybrid search, and injects the top-15 matches as a clearly labeled "RETRIEVAL CONTEXT" block into the system prompt. This grounds the LLM's recommendations in specific catalog items rather than relying on training-time knowledge. The full catalog listing (name | type | levels | duration | URL) is also permanently in the system prompt as a reference layer — the LLM can "see" everything, and retrieval highlights the most relevant candidates.

**Vague-query guard:** A regex pattern bank detects first-turn messages that are too sparse to act on ("I need an assessment", "help me hire"). A post-generation safety net additionally clears any recommendations if the guard fires, regardless of what the model said.

**Grounding validation:** After every LLM response is parsed, each recommendation is cross-checked: the URL must exist in the catalog set, and the name must match a catalog entry (exact or partial). Hallucinated items are silently dropped and a warning is logged. This provides a hard safety net independent of the prompt.

---

## 3. Agent Design — When to Ask, Retrieve, Answer, Refuse

| Condition | Action |
|---|---|
| First turn, vague or greeting | Ask 1–2 targeted clarifying questions; no recommendations |
| Sufficient context (role + level or description) | Retrieve → recommend 1–10 items with justification |
| User adds constraints mid-conversation | Update shortlist in-place, acknowledge what changed |
| Comparison question | Answer from catalog metadata only (duration, type, levels, description) |
| Off-topic / legal / competitor / injection | Fast-path refuse before API call; empty recommendations |
| Turn count approaches 8 | Agent is prompted to consolidate and commit to a shortlist |

The system prompt enforces JSON output (`{"reply", "recommendations", "end_of_conversation"}`). Temperature=0.3 makes format compliance highly reliable; the fallback regex JSON extractor catches edge cases where the model includes prose around the JSON.

---

## 4. Evaluation Approach

Ten conversation traces cover the full behavioral surface:

- **Happy path:** Java mid-level, senior data scientist, graduate scheme, executive leadership
- **Clarification:** Vague first turn must produce a question, not recommendations
- **Refinement:** Adding personality constraints mid-conversation updates shortlist
- **Comparison:** OPQ32 vs GSA factual comparison
- **Refusal:** EEOC question, prompt injection attempt
- **Volume hiring:** Retail frontline with duration constraint (≤20 min)

Each trace is evaluated on: schema compliance, URL/name grounding (anti-hallucination), behavioral assertions (e.g. "contains_personality_test"), and **Recall@10** against a labeled expected shortlist.

**What didn't work initially:**
- Pure semantic search ranked generic ability tests too high for technical roles because the embeddings encode concept similarity but miss the specificity of "Java developer" → "Java 8 (New)". Keyword boosting fixed this.
- The LLM would occasionally recommend on turn 1 for borderline-vague queries. The vague-query regex guard + post-generation safety net eliminated this.
- Early prompts using prose format led to occasional JSON parsing failures. Switching to strict JSON-only output with a fallback regex extractor solved reliability.

**Measured improvement:** After adding hybrid retrieval over pure semantic search, Recall@10 on technical role traces improved from ~0.5 to ~0.8 in local testing. The grounding validator caught 2–3 hallucinated URLs in early iterations before the system prompt was tightened.

---

## 5. Stack Justification

| Component | Choice | Why |
|---|---|---|
| LLM | Claude Sonnet (claude-sonnet-4-20250514) | Best instruction-following and JSON reliability; Anthropic SDK available |
| Embeddings | all-MiniLM-L6-v2 | 22M params, fast cold-start, excellent semantic quality for short texts |
| Vector store | FAISS IndexFlatIP | Zero infra, exact cosine similarity, 50-item catalog trivially fits in RAM |
| API framework | FastAPI + Pydantic v2 | Native async, automatic schema validation, OpenAPI docs free |
| Deployment | Render (render.yaml) + Docker | Free tier, cold-start ≤2 min matches evaluator's /health allowance |

**AI tools used:** Claude assisted with boilerplate structuring. All design decisions, architectural choices, retrieval strategy, guard logic, and evaluation design are my own and I can defend each in a technical interview.

---

*Total lines of code: ~800 across 5 Python files. Test coverage: 10 conversation traces, 30+ behavioral assertions.*
