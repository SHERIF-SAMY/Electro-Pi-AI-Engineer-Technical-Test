# 📋 Example Q&A — RAG Pipeline Outputs

This file documents **3 representative example questions** and the actual answers
produced by the RAG pipeline, demonstrating retrieval quality, citation accuracy,
and hallucination prevention.

> **Document used for testing:** A short generated PDF document (`rag_paper.pdf`) about Retrieval-Augmented Generation.
>
> **Model stack:** `nomic-embed-text` (Ollama) for embeddings, `llama-3.3-70b-versatile`
> (Groq) for generation, FAISS with `similarity_score_threshold=0.4`, `k=4`.

---

## Example 1 — On-topic, factual question

**Question:**
> *"What is Retrieval-Augmented Generation and what problem does it solve?"*

**Answer produced by the pipeline:**
```
**Retrieval-Augmented Generation (RAG)** is a hybrid AI architecture that combines parametric memory with non-parametric memory. 
According to the context, RAG **solves two main problems**: 
* **Knowledge staleness**
* **Hallucination** 
by grounding answers in retrieved documents.
```

**Citations returned:**
```
[1] 📄 rag_paper.pdf | Page 1 | Chunk 1/1 | Score: 0.506
    Preview: "Retrieval-Augmented Generation (RAG) is a hybrid AI architecture that combines parametric
memory with non-parametric memory. It solves knowledge staleness and hallucination by grounding
answers in ret"
```

**Assessment notes:**
- Correct answer grounded entirely in the document
- Citations are precise (filename, page, chunk, score, preview)

---

## Example 2 — Multi-hop reasoning question

**Question:**
> *"What retrieval model does RAG use, and how does it compare to BM25?"*

**Answer produced by the pipeline:**
```
**RAG uses Dense Passage Retrieval (DPR)** as its retrieval component. 
According to the context, DPR **provides substantial improvements** compared to BM25 retrieval, because dense retrieval can match **semantically equivalent but lexically different passages**.
```

**Citations returned:**
```
[1] 📄 rag_paper.pdf | Page 1 | Chunk 1/1 | Score: 0.591
    Preview: "Retrieval-Augmented Generation (RAG) is a hybrid AI architecture that combines parametric
memory with non-parametric memory. It solves knowledge staleness and hallucination by grounding
answers in ret"
```

**Assessment notes:**
- Answer synthesizes details accurately from the context chunk.
- Citation includes the accurate similarity score (> 0.4 threshold).

---

## Example 3 — Hallucination prevention ("No Relevant Context Found")

**Question:**
> *"What is the capital city of France, and what is the population of Paris?"*

**Answer produced by the pipeline:**
```
I could not find any relevant information in the uploaded document to answer your question. Please try rephrasing, or ask something that is covered in the document.
```

**Citations returned:** `null`

**`no_context_found` flag:** `true`

**What happened (internal pipeline):**
1. Query was embedded and FAISS searched the index for the top-4 similar chunks
2. All retrieved chunks scored **below the 0.4 similarity threshold**
3. The retriever returned an empty list (no chunks passed the threshold)
4. The LLM was **never called** — the system returned the fallback response
   directly from `chat_service.py` without spending any API tokens
5. The `no_context_found: true` flag was set in the API response

**Why this matters:**
Without the similarity score threshold, FAISS would return irrelevant chunks as
"context" — then the LLM would likely produce a hallucinated answer about Paris population using its
parametric knowledge, violating the RAG contract. The threshold prevents this at
the infrastructure level.

---

## Summary Table

| # | Question Type | Retrieved Chunks | Score Range | Hallucinated? |
|---|---|---|---|---|
| 1 | Factual definition | 1 | ~0.506 | No |
| 2 | Comparative reasoning | 1 | ~0.591 | No |
| 3 | Off-topic (OOD) | 0 (below threshold) | < 0.40 | No — blocked at infrastructure layer |
