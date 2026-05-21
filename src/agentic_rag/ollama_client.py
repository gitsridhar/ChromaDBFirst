from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class OllamaModel:
    name: str
    size: int | None


def list_models(base_url: str, timeout_s: int = 30) -> list[OllamaModel]:
    response = requests.get(f"{base_url}/api/tags", timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()

    models = payload.get("models", [])
    return [
        OllamaModel(name=str(model.get("name", "")), size=model.get("size"))
        for model in models
        if model.get("name")
    ]


def list_model_names(base_url: str) -> list[str]:
    return [m.name for m in list_models(base_url=base_url)]


def _is_embedding_model_name(name: str) -> bool:
    lower = name.lower()
    keywords = ["embed", "embedding", "bge", "e5", "minilm", "nomic"]
    return any(keyword in lower for keyword in keywords)


def resolve_model_name(base_url: str, requested: str, model_type: str) -> str:
    names = list_model_names(base_url=base_url)
    if not names:
        raise ValueError("No models found in Ollama. Run 'ollama pull <model>' first.")

    if requested != "auto":
        if requested not in names:
            available = ", ".join(names)
            raise ValueError(
                f"Model '{requested}' not found in Ollama. Available models: {available}"
            )
        return requested

    if model_type == "embedding":
        for name in names:
            if _is_embedding_model_name(name):
                return name
        raise ValueError(
            "Could not auto-detect an embedding model. Pass --embedding-model explicitly."
        )

    for name in names:
        if not _is_embedding_model_name(name):
            return name

    # If everything looks like embeddings, fall back to the first model.
    return names[0]


def chat_with_ollama(base_url: str, model: str, system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()

    message = payload.get("message", {})
    content = message.get("content", "").strip()
    if not content:
        raise ValueError("No answer content returned by Ollama /api/chat")
    return content
