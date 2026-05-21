# Agentic AI RAG App (Ollama + ChromaDB + PDF)

This project is a local-first agentic RAG application that:
- reads PDF files,
- chunks and embeds text using a local Ollama embedding model,
- stores vectors in ChromaDB,
- answers questions with a local Ollama chat model using retrieved context.

## 1) Prerequisites

- macOS with Python 3.10+
- Ollama installed and running:

```bash
ollama serve
```

Pull local models (example defaults used by this app):

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

List all models currently exposed by Ollama:

```bash
/Users/sridharvenkat/Documents/chromadbollama/.venv/bin/python -m agentic_rag models
```

## 2) Install

From project root:

```bash
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 3) Optional Environment Variables

Defaults are fine for local usage.

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `CHROMA_DB_PATH` (default: `chroma_db`)

Example:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export CHROMA_DB_PATH=chroma_db
```

## 4) Ingest PDF into Vector DB

```bash
agentic-rag ingest \
  --pdf /absolute/path/to/your/file.pdf \
  --collection pdf_docs \
  --embedding-model auto
```

## 5) Ask Questions

```bash
agentic-rag ask \
  --question "What are the key points in this PDF?" \
  --collection pdf_docs \
  --chat-model auto \
  --embedding-model auto \
  --top-k 4
```

To force specific models, replace `auto` with any installed Ollama model name.

## 6) General NLP Task Over Retrieved Context

```bash
agentic-rag nlp \
  --task "Summarize the compliance obligations as bullet points" \
  --query "compliance obligations and requirements" \
  --collection pdf_docs \
  --chat-model auto \
  --embedding-model auto \
  --top-k 6
```

## 7) How It Works

- PDF extraction: `pypdf`
- Chunking: word-based chunks with overlap
- Embeddings: Ollama `/api/embed`
- Vector store: ChromaDB persistent collection
- Generation: Ollama `/api/chat` with retrieved context injected into prompt

## 8) Notes

- Re-ingesting the same PDF skips existing chunk IDs.
- Chroma data persists under `chroma_db/`.
- If retrieval is weak, try:
  - smaller chunk size,
  - higher `--top-k`,
  - different embedding model.
# ChromaDBFirst
