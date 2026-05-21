from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .config import get_settings
from .ollama_client import list_models, resolve_model_name
from .pdf_ingest import extract_pdf_chunks, ollama_embed
from .rag_app import answer_question, run_nlp_task
from .vector_store import ChromaVectorStore


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    embedding_model = resolve_model_name(
        base_url=settings.ollama_base_url,
        requested=args.embedding_model,
        model_type="embedding",
    )
    store = ChromaVectorStore(collection_name=args.collection)

    chunks = list(extract_pdf_chunks(args.pdf))
    if not chunks:
        raise ValueError(f"No extractable text found in PDF: {args.pdf}")

    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embeddings = [
        ollama_embed(
            base_url=settings.ollama_base_url,
            model=embedding_model,
            text=text,
        )
        for text in texts
    ]

    existing_ids = set(store.list_ids())
    filtered = [
        (chunk_id, text, embedding, metadata)
        for chunk_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)
        if chunk_id not in existing_ids
    ]

    if not filtered:
        print("No new chunks to add. All chunk IDs already exist in this collection.")
        return

    f_ids, f_texts, f_embeddings, f_metas = map(list, zip(*filtered))
    store.add_chunks(
        chunk_ids=f_ids,
        texts=f_texts,
        embeddings=f_embeddings,
        metadatas=f_metas,
    )

    print(f"Ingested {len(f_ids)} chunks from {Path(args.pdf).name}")
    print(f"Embedding model used: {embedding_model}")
    print(f"Collection '{args.collection}' now has {store.count()} chunks")


def cmd_ask(args: argparse.Namespace) -> None:
    settings = get_settings()
    chat_model = resolve_model_name(
        base_url=settings.ollama_base_url,
        requested=args.chat_model,
        model_type="chat",
    )
    embedding_model = resolve_model_name(
        base_url=settings.ollama_base_url,
        requested=args.embedding_model,
        model_type="embedding",
    )

    result = answer_question(
        collection_name=args.collection,
        question=args.question,
        chat_model=chat_model,
        embedding_model=embedding_model,
        top_k=args.top_k,
    )
    print("\nAnswer:\n")
    print(result.answer)
    print(f"\nChat model: {chat_model}")
    print(f"Embedding model: {embedding_model}")

    print("\nRetrieved Context:\n")
    for idx, chunk in enumerate(result.context_chunks, start=1):
        print(
            f"[{idx}] source={chunk.source}, page={chunk.page}, "
            f"distance={chunk.distance:.4f}, chunk_id={chunk.chunk_id}"
        )


def cmd_models(_: argparse.Namespace) -> None:
    settings = get_settings()
    models = list_models(base_url=settings.ollama_base_url)
    if not models:
        print("No models found in Ollama. Pull a model first: ollama pull <model>")
        return

    print("Installed Ollama models:\n")
    for model in models:
        size_mb = (model.size / (1024 * 1024)) if model.size else None
        size_text = f" ({size_mb:.1f} MB)" if size_mb else ""
        print(f"- {model.name}{size_text}")


def cmd_nlp(args: argparse.Namespace) -> None:
    settings = get_settings()
    chat_model = resolve_model_name(
        base_url=settings.ollama_base_url,
        requested=args.chat_model,
        model_type="chat",
    )
    embedding_model = resolve_model_name(
        base_url=settings.ollama_base_url,
        requested=args.embedding_model,
        model_type="embedding",
    )

    result = run_nlp_task(
        collection_name=args.collection,
        task=args.task,
        query=args.query,
        chat_model=chat_model,
        embedding_model=embedding_model,
        top_k=args.top_k,
    )

    print("\nNLP Result:\n")
    print(result.answer)
    print(f"\nChat model: {chat_model}")
    print(f"Embedding model: {embedding_model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-rag",
        description="Local Ollama + ChromaDB RAG app for PDF question answering.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Read PDF, embed, and store in ChromaDB.")
    ingest.add_argument("--pdf", required=True, help="Path to the PDF file")
    ingest.add_argument("--collection", default="pdf_docs", help="Chroma collection name")
    ingest.add_argument(
        "--embedding-model",
        default="auto",
        help="Ollama embedding model name, or 'auto'",
    )
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a question using RAG over stored chunks.")
    ask.add_argument("--question", required=True, help="Question for the assistant")
    ask.add_argument("--collection", default="pdf_docs", help="Chroma collection name")
    ask.add_argument(
        "--chat-model",
        default="auto",
        help="Ollama chat model name, or 'auto'",
    )
    ask.add_argument(
        "--embedding-model",
        default="auto",
        help="Ollama embedding model for retrieval, or 'auto'",
    )
    ask.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    ask.set_defaults(func=cmd_ask)

    models = sub.add_parser("models", help="List models exposed by local Ollama.")
    models.set_defaults(func=cmd_models)

    nlp = sub.add_parser("nlp", help="Run a general NLP task over retrieved PDF context.")
    nlp.add_argument("--task", required=True, help="NLP task instruction")
    nlp.add_argument("--query", required=True, help="Query used for embedding-based retrieval")
    nlp.add_argument("--collection", default="pdf_docs", help="Chroma collection name")
    nlp.add_argument(
        "--chat-model",
        default="auto",
        help="Ollama chat model name, or 'auto'",
    )
    nlp.add_argument(
        "--embedding-model",
        default="auto",
        help="Ollama embedding model for retrieval, or 'auto'",
    )
    nlp.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    nlp.set_defaults(func=cmd_nlp)

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
