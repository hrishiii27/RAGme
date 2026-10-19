# RAGme — Questions for You

> Write your answer under each question (`→ ...`). If you're unsure, write "go with your rec" and I'll use my recommended option (★).
> Context: [memory.md](./memory.md)

---

## A. Learning style & scope
**A1. How hands-on do you want to be?**
(a) I write the code and you guide/review · (b) ★ You scaffold, I implement the core RAG pieces with your help · (c) You build it and explain each part
→ (b)

**A2. Timeline / pace?** (e.g. a weekend sprint, a few weeks, open-ended)
→ a weekend sprint

**A3. Final deliverable?** A local-only learning project, a portfolio piece (GitHub + demo), or actually deployed for real users?
→ portfolio piece (GitHub + demo)

## B. Data scope
The corpus is **65k docs / ~1 GB text / ~250M tokens**. Embedding all of it many times over (once per chunking × embedding combo) is slow and/or expensive.

**B1. Corpus size for experiments?**
★ Run experiments on a **stratified dev subset (~3–5k docs)** and index the full corpus only with the winning config · or use the full corpus from the start
→ Run experiments on a **stratified dev subset (~5–8k docs)** and index the full corpus only with the winning config · or use the full corpus from the start

**B2. Indic-language documents (~14%) and noisy OCR (~11%)?**
(a) English-only for now · (b) ★ Keep them and evaluate multilingual embeddings as one of the comparisons · (c) Full multilingual, including Hindi queries
→ (b)

**B3. Should users be able to ask in Hindi/regional languages and get answers in the same language?**
→ (c) (but lets focus on one language for now, English then we shall evaluate the efficiency of that and then pivot into multilingual)

**B4. Any topic focus?** (e.g. schemes & benefits, law/judiciary, policy, universities) or all 36 topics?
→ ★ All 36 topics

## C. Compute & budget
**C1. Your machine?** (Apple Silicon chip + RAM, e.g. M2 Pro 16 GB) Any GPU access (Colab, cloud)?
→ Macbook Air M2 with 8GB RAM

**C2. API budget?** ($0 local-only · ~$5–20 · more). Which API keys do you have? (OpenAI, Gemini, Anthropic, Cohere, Voyage, Groq, HF…)
→ I have access to Gemini API key, i can get a few open-source ones form openrouter

**C3. LLM for answer generation?**
(a) API model (Gemini / GPT / Claude) · (b) Local via Ollama (Llama / Qwen / Gemma) · (c) ★ Both: a local one for cheap iteration and an API one for the final quality comparison
→ (c)

## D. Stack preferences
**D1. Vector DB?**
★ **Qdrant** (local Docker; dense + sparse hybrid built in, good metadata filtering) · Chroma (simplest) · pgvector (Postgres) · LanceDB · FAISS (raw, very educational) · no preference
→ Lets start with Chroma, i'd also want to learn Docker along the way so we can decide based off of that

**D2. Framework?**
(a) ★ **Mostly from scratch** (plain Python + small libraries) so you see every phase, using LangChain/LlamaIndex only for specific splitters · (b) LlamaIndex · (c) LangChain/LangGraph · (d) no preference
→ (a) i'd also want to learn about langchain in the process

**D3. Frontend framework** (design comes later, but this affects scaffolding)?
Vanilla HTML/JS · ★ React (Vite) · Next.js · no preference
→ React (Vite)

**D4. Eval tooling?** ★ RAGAS + a custom harness, DeepEval, TruLens, Arize Phoenix, LangSmith, or no preference
→ Ragas

## E. Agents & product behaviour
**E1. What do you mean by "agent orchestration" here?** Pick any:
- Query routing (e.g. topic/state classifier → filtered search)
- Query decomposition for multi-part questions
- Agentic / iterative retrieval (the LLM decides to search again)
- Tool use (e.g. link to the official site, eligibility checker for schemes)
- Just a single-shot RAG pipeline for now, with agents later ★
→ Just a single-shot RAG pipeline for now, with agents later (but we can keep in mind for the future)

**E2. Multi-turn chat with conversation memory**, or single-question Q&A?
→ Multi-turn chat with conversation memory

**E3. Who are the target users?** (citizens, students, researchers, government employees…). This shapes the system prompt tone.
→ General public

**E4. When the answer isn't in the corpus**, should the bot refuse ("I couldn't find this in official sources") or fall back to general LLM knowledge with a disclaimer? ★ Refuse + suggest where to look
→ ★ Refuse + suggest where to look. 

**E5. Is user auth / chat history storage needed?**
→ No, not right now. 

## F. Evaluation
**F1. Golden eval set:** OK with ★ **LLM-generated synthetic Q&A from chunks, then you hand-review a subset**, plus ~30–50 questions you write yourself as real-user questions?
→ yes, i'd like to get around 10 basic questions then grade them on input, model response, model critique, model outcome, human critique, human outcome, alignment and agreement so that i create a general skeleton for our prompts and evaluate them with that, we can later expand this set with synthetic questions and desired behaviour learned from the initial file and can expand on that. 

**F2. Human Model Eval:** who are the raters (just you, friends, colleagues)? Roughly how many judgements can you realistically do (50? 200?)
→ just me. around 50. 

**F3. Which quality dimensions matter most to you?** Rank them: correctness · faithfulness (no hallucination) · citation quality · completeness · conciseness · latency · cost
→ correctness, faithfulness, citation quality, completeness. (latency and cost are secondary priority). Collecting examples and categorizing failure models is useful could learn langfuse along the way

## G. Anything else
**G1. Anything you've already read or tried** (courses, papers, tools) that I should build on or avoid repeating?
→ i've tried langGraph and it's okay i guess, but i didn't like it much cause it felt too complex for no reason. 

**G2. Repo conventions?** (Git + GitHub? Python version? Tests required?)
→ Git + GitHub, python 3.10+ and we'll do unit tests for now and as the project expands we can scale it up. 
