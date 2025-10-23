# RAG Mini-Project

A simple progression from a baseline, window-based RAG to HYDE (hypothetical answer–guided retrieval), and finally multi-query expansion with rank fusion.

- **Goal:** study → build → compare.
- **Data:** local PDFs/text split into chunks, embedded, and searched with ChromaDB.
- **Models/Tools:** ChromaDB, OpenAI for generation where needed, SentenceTransformer embeddings, LangChain text splitters, optional UMAP plots for intuition.

---

## Why three steps?

1. **Baseline (deterministic windows):** get something working and measurable.
2. **HYDE:** generate a short hypothetical answer to the query; use that to retrieve semantically closer chunks.
3. **Query Expansion:** generate multiple query variants and fuse their rankings to reduce single-query blind spots.

Each step stands alone but builds on the last.

---

## File-by-file

### 1) `app.py` — Baseline windowed RAG
- **What it does:**  
  - Loads documents, **splits into fixed-size overlapping chunks** (sliding window).
  - **Embeds** chunks (OpenAI embedding via Chroma’s `OpenAIEmbeddingFunction`).
  - **Indexes** into a **persistent Chroma** DB (`chroma_persistent_storage`).
  - **Retrieves** top-k chunks for a user question.
  - **Answers** with an OpenAI chat model using the retrieved context.
- **Why it’s here:** Establish a working reference: deterministic chunking + single query → retrieve → answer.
- **Notes:** Chunk size and overlap are explicit in the splitter; the script includes a sample question to run end-to-end.

---

### 2) `expansion_answer.py` — HYDE: retrieve with a hypothetical answer
- **What it does:**  
  - **Reads source data** (e.g., PDFs) and prepares text.  
  - Uses **LangChain text splitters** (`RecursiveCharacterTextSplitter`, `SentenceTransformersTokenTextSplitter`) to avoid hard cuts mid-sentence/token.  
  - **Embeds** chunks with a **SentenceTransformer** via Chroma (local, no API calls for embeddings).  
  - **Generates a hypothetical answer** to the user query with an LLM (HYDE).  
  - Forms a **joint query** (original + hypothetical answer) and **retrieves** with that.  
  - Includes **UMAP projections** (optional) to visualize how the enhanced query moves in embedding space relative to the corpus.
  - (Optional) **Final answer generation** from retrieved chunks.
- **Why it’s here:** HYDE often pulls in more on-topic passages than the raw query when the query is sparse or underspecified.
- **What to look at:** The shift from character-level to token-aware splitting; the creation and use of the **joint query**; UMAP to sanity-check the semantic move.

---

### 3) `expansion_queries.py` — Multi-query expansion + Reciprocal Rank Fusion (RRF)
- **What it does:**  
  - **Generates multiple query variants** from the original query (paraphrases / related sub-queries).  
  - Splits text (character → token-aware) and **embeds** with a SentenceTransformer via Chroma.  
  - **Retrieves top-k** for **each** query variant.  
  - **Fuses rankings** using **RRF** (`1/(K + rank)`) to get a robust final list.  
  - Deduplicates and prints **top fused results** (doc id + score).  
  - Optional **UMAP** code (commented) for plotting queries vs. docs.
- **Why it’s here:** Single queries can miss relevant pockets of text. Query expansion + RRF improves recall while keeping noise in check.
- **What to look at:**  
  - The **rank consolidation** utilities (`build_rankings`, `rrf`, `unique_docs_from`).  
  - How multiple queries diversify retrieval and how **RRF** balances them.

---

## Typical flow

1. **Baseline:** run `app.py` → confirm ingest, indexing, and retrieval work end-to-end.  
2. **HYDE:** run `expansion_answer.py` → compare retrieved chunks vs. baseline for the same question.  
3. **Query Expansion:** run `expansion_queries.py` → generate variants, fuse, and compare the final set of chunks.

---

## Setup (minimal)

- Python 3.10+
- `pip install chromadb openai sentence-transformers langchain pypdf umap-learn numpy`
- Set your OpenAI key if using LLM calls:  
  ```bash
  export OPENAI_API_KEY=your_key_here
