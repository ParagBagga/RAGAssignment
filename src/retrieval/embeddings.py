import json
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

CHUNKS_FILE = Path("data/processed/chunks.json")
OUTPUT_FILE = Path("data/processed/embedded_chunks.json")


def create_embedding(text: str):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"][0]


if __name__ == "__main__":

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Total chunks: {len(chunks)}")

    embedded_chunks = []

    for index, chunk in enumerate(chunks):

        print(f"Embedding chunk {index + 1}/{len(chunks)}")

        embedding = create_embedding(chunk["text"])

        embedded_chunks.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "embedding": embedding
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(embedded_chunks, file)

    print("\nFinished!")
    print(f"Saved to: {OUTPUT_FILE}")