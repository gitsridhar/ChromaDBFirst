from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

from pypdf import PdfReader
import requests


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_chunks(pdf_path: str | Path) -> Iterator[TextChunk]:
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    for page_idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        clean = normalize_whitespace(raw)
        if not clean:
            continue

        for i, chunk in enumerate(chunk_text(clean), start=1):
            yield TextChunk(
                chunk_id=f"{pdf_path.stem}-p{page_idx}-c{i}",
                text=chunk,
                metadata={
                    "source": pdf_path.name,
                    "page": page_idx,
                },
            )


def ollama_embed(base_url: str, model: str, text: str, timeout_s: int = 120) -> list[float]:
    response = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": text},
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()

    embeddings = payload.get("embeddings", [])
    if not embeddings:
        raise ValueError("No embeddings returned from Ollama /api/embed")
    return embeddings[0]
