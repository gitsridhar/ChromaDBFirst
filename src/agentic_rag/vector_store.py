from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import chromadb
from chromadb.api.models.Collection import Collection

from .config import get_settings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page: int
    chunk_id: str
    distance: float


class ChromaVectorStore:
    def __init__(self, collection_name: str) -> None:
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self.collection.add(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc, meta, distance, chunk_id in zip(docs, metas, distances, ids):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=str(meta.get("source", "unknown")),
                    page=int(meta.get("page", -1)),
                    chunk_id=str(chunk_id),
                    distance=float(distance),
                )
            )
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def list_ids(self) -> Iterable[str]:
        result = self.collection.get(include=[])
        return result.get("ids", [])
