from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from .config import get_settings
from .ollama_client import chat_with_ollama
from .pdf_ingest import ollama_embed
from .vector_store import ChromaVectorStore, RetrievedChunk


@dataclass(frozen=True)
class DirectNLPResult:
    answer: str
    model: str
    task: str


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    context_chunks: list[RetrievedChunk]


def build_context(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] source={chunk.source}, page={chunk.page}, chunk_id={chunk.chunk_id}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def answer_question(
    collection_name: str,
    question: str,
    chat_model: str,
    embedding_model: str,
    top_k: int = 4,
) -> RAGAnswer:
    settings = get_settings()
    store = ChromaVectorStore(collection_name=collection_name)

    q_embedding = ollama_embed(
        base_url=settings.ollama_base_url,
        model=embedding_model,
        text=question,
    )

    chunks = store.query(embedding=q_embedding, top_k=top_k)
    if not chunks:
        raise ValueError("No chunks found in vector store. Please ingest a PDF first.")

    context = build_context(chunks)
    system_prompt = dedent(
        """
        You are a helpful RAG assistant.
        Answer strictly from the provided context.
        If context is insufficient, say you do not know.
        Include source references in the final answer using [n] markers.
        """
    ).strip()

    user_prompt = dedent(
        f"""
        Question:
        {question}

        Context:
        {context}
        """
    ).strip()

    answer = chat_with_ollama(
        base_url=settings.ollama_base_url,
        model=chat_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return RAGAnswer(answer=answer, context_chunks=chunks)


def run_nlp_task(
    collection_name: str,
    task: str,
    query: str,
    chat_model: str,
    embedding_model: str,
    top_k: int = 4,
) -> RAGAnswer:
    settings = get_settings()
    store = ChromaVectorStore(collection_name=collection_name)

    q_embedding = ollama_embed(
        base_url=settings.ollama_base_url,
        model=embedding_model,
        text=query,
    )
    chunks = store.query(embedding=q_embedding, top_k=top_k)
    if not chunks:
        raise ValueError("No chunks found in vector store. Please ingest a PDF first.")

    context = build_context(chunks)
    system_prompt = dedent(
        """
        You are an NLP assistant.
        Perform the user task using only the provided context.
        If context is insufficient, state that clearly.
        Cite evidence with [n] markers from context blocks.
        """
    ).strip()

    user_prompt = dedent(
        f"""
        NLP Task:
        {task}

        Query:
        {query}

        Context:
        {context}
        """
    ).strip()

    answer = chat_with_ollama(
        base_url=settings.ollama_base_url,
        model=chat_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return RAGAnswer(answer=answer, context_chunks=chunks)


def run_direct_nlp(
    task: str,
    text: str,
    chat_model: str,
) -> DirectNLPResult:
    """Run an NLP task directly on supplied text without any vector retrieval."""
    settings = get_settings()

    system_prompt = dedent(
        """
        You are an expert NLP assistant.
        Perform the requested task accurately on the provided text.
        Only use information present in the text.
        """
    ).strip()

    user_prompt = dedent(
        f"""
        NLP Task:
        {task}

        Text:
        {text}
        """
    ).strip()

    answer = chat_with_ollama(
        base_url=settings.ollama_base_url,
        model=chat_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return DirectNLPResult(answer=answer, model=chat_model, task=task)
