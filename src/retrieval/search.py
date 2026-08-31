import json
from pathlib import Path

import faiss
import numpy as np
import requests


OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

INDEX_FILE = Path("data/processed/faiss_index.bin")
CHUNKS_FILE = Path("data/processed/embedded_chunks.json")


def create_embedding(text: str):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        }
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def search(query, top_k=4):

    # Load FAISS index
    index = faiss.read_index(str(INDEX_FILE))

    # Load original chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Convert user question into embedding
    query_embedding = create_embedding(query)

    # FAISS expects a 2D float32 array
    query_vector = np.array(
        [query_embedding]
    ).astype("float32")

    # Search FAISS
    distances, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    for distance, chunk_index in zip(
        distances[0],
        indices[0]
    ):

        chunk = chunks[chunk_index]

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":

    query = "How does self-attention work?"

    results = search(query)

    print(f"\nQuestion: {query}\n")

    for i, result in enumerate(results, start=1):

        print(f"--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Distance: {result['distance']}")
        print(f"\nText:\n{result['text'][:500]}")
        print()