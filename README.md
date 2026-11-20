<h1 align="center">Lunar Research Desk</h1>
<p align="center">Ask a question about the Moon. Inspect the NASA passage behind each answer.</p>
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-166534">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-4169E1">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-B42318">
  <img alt="Groq" src="https://img.shields.io/badge/Groq-A54100">
</p>

## Try a question

Ask **“How much rock and soil did astronauts bring back to Earth?”** in the Streamlit chat. Each question automatically retrieves four passages and asks Groq for an answer with supporting quotations and PDF page links. Expand the passages and follow a page link to check the wording in the original PDF. Chat history is visible during the browser session. Phrase each question independently because the retriever and model receive only the current turn.

## Architecture

~~~mermaid
flowchart TD
    PDF[Two NASA PDFs] --> P[Extract text by physical PDF page]
    P --> C[Split into overlapping passages]
    C --> E[BGE embeddings]
    C --> DB[(PostgreSQL passages and page metadata)]
    E --> DB
    Q[Question from Streamlit or CLI] --> L[Postgres full-text search]
    Q --> D[BGE query embedding and cosine search]
    DB --> L
    DB --> D
    L -->|top 20| F[Reciprocal rank fusion]
    D -->|top 20| F
    F -->|top 16| R[MiniLM cross-encoder reranker]
    R -->|top 4| UI[Passage viewer]
    R -->|top 4| G[LangChain prompt and Groq model]
    G --> V[Validate passage number and quote]
    V --> A[Answer with PDF page links or abstention]
~~~

| Stage | What happens and why |
|---|---|
| 1. Extract | pypdf reads each physical page of both publications. Keeping the page number allows a direct link back to the evidence. |
| 2. Chunk | LangChain recursively splits each page into passages of up to 850 characters with up to 120 characters of overlap. This keeps most nearby context together. |
| 3. Embed and store | BGE-small-en-v1.5 turns each passage into a normalized 384-number vector. PostgreSQL stores the vector, passage, source ID and page. It generates an English full-text column with a GIN index. Re-ingestion replaces the project's passage rows in one transaction. |
| 4. Retrieve twice | PostgreSQL finds up to 20 lexical matches. For semantic search, the app embeds the question with the BGE retrieval instruction and takes the top 20 cosine similarities. Both branches read the same database snapshot. |
| 5. Fuse | Each occurrence in a ranked list contributes 1 / (60 + rank). Passages that rank well in both lists accumulate points. The best 16 advance. |
| 6. Rerank | MiniLM reads the question and each candidate together. Its relevance score determines the final four passages. This extra comparison happens only on the shortlist. |
| 7. Generate | In answer mode, LangChain fills the prompt with the question and numbered passages. Groq returns claims with numbered evidence IDs. Python copies each selected source sentence directly from the retrieved text. Credentials stay in the local environment. |
| 8. Validate and display | For each claim, Python checks the passage number and verifies that the normalized quotation appears in that passage and contains at least 12 characters. It constructs the PDF link from trusted source metadata. Empty claims or a failed citation check yield an insufficient-evidence message. |

## Verification and architectural limits

On 20 November 2026, the 400-passage index returned a reference-pattern match at rank one for all four evaluation questions. Evidence hit@4 and MRR@4 were both 1.00. Generated answers matched the expected text pattern on **3 of 4** questions. The unrelated Wi-Fi question triggered abstention. All four offline citation tests passed. Hosted output can vary between runs.

- **Citation quality.** A successful claim includes a quotation and a physical PDF page link. An exact quotation can still be interpreted incorrectly. The validator checks provenance, not whether every conclusion follows from the quotation.
- **False abstention.** One malformed citation discards the whole response. An earlier live regolith question retrieved four passages but produced an abstention. The interface gives the same message for insufficient evidence and failed validation, making those failures hard to distinguish.
- **Retrieval scale.** Dense search transfers every passage and vector from PostgreSQL to Python for every question. This is workable for 400 passages. A larger corpus would benefit from indexed vector search inside PostgreSQL.
- **Lexical recall.** The English plain-text query requires all surviving search terms. A verbose question can miss relevant lexical matches. Compare lexical-only, dense-only and hybrid results before tuning fusion or reranking.
- **PDF layout.** Page-local chunking can split an explanation across pages. Multi-column text and tables can lose reading order during extraction.
- **Evaluation coverage.** Four factual questions and one unrelated question give a repeatable smoke test. Regex matching can miss paraphrases or reward unsupported wording. A stronger evaluation would label relevant pages, judge each claim against evidence, and include ambiguous and multi-turn questions.

## Run locally

Use Python, a running PostgreSQL instance and a database role with permission to create a table and index. Model weights download on first use. Retrieval runs locally. Hosted answers require a Groq key and an enabled model that supports tool calling.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Set PGHOST, PGPORT, PGDATABASE, PGUSER and PGPASSWORD in .env. For hosted answers set GROQ_API_KEY and GROQ_MODEL. The tested model was openai/gpt-oss-20b. Keep .env local.

~~~powershell
.\.venv\Scripts\python.exe app.py ingest
.\.venv\Scripts\python.exe -m streamlit run frontend.py --server.address 127.0.0.1
~~~

Open the local URL printed by Streamlit. For CLI use or evaluation:

~~~powershell
.\.venv\Scripts\python.exe app.py search "What is lunar regolith?"
.\.venv\Scripts\python.exe app.py ask "How much lunar material was brought back?"
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe evaluate.py --answers
~~~

## Repository guide

| File | Responsibility |
|---|---|
| corpus.py | Source manifest, PDF extraction, chunking and transactional indexing |
| search.py | Lexical and dense retrieval, fusion and reranking |
| generation.py | LangChain prompt, Groq call, citation checks and rendering |
| frontend.py / app.py | Streamlit chat and command-line entry points |
| evaluate.py / test_citations.py | Live evaluation and deterministic citation tests |
| data/ | The two NASA PDFs used to build the index |

## Sources and models

- [NASA Moon lithograph](https://science.nasa.gov/wp-content/uploads/2024/01/62217main-moon-lithograph.pdf)
- [NASA Exploring the Moon teacher guide](https://science.nasa.gov/wp-content/uploads/2024/01/exploring-the-moon-teachers-guide.pdf)
- [BGE embedding model](https://huggingface.co/BAAI/bge-small-en-v1.5) and [MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

## About me

Maintained by [Danyal-0276](https://github.com/Danyal-0276). This portfolio project explores how to retrieve evidence from public science documents and make generated answers easier to check.

**Project topics** · Python · LangChain · retrieval-augmented generation · hybrid search · reranking · PostgreSQL · Streamlit · lunar science
