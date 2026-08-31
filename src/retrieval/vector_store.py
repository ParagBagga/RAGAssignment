import json
from pathlib import Path

import faiss
import numpy as np


INPUT_FILE = Path("data/processed/embedded_chunks.json")
INDEX_FILE = Path("data/processed/faiss_index.bin")


def create_faiss_index():

    # Load embedded chunks
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        embedded_chunks = json.load(file)

    print(f"Total embedded chunks: {len(embedded_chunks)}")

    # Extract only embeddings
    embeddings = []

    for chunk in embedded_chunks:
        embeddings.append(chunk["embedding"])

    # Convert Python list to NumPy array
    embeddings = np.array(embeddings).astype("float32")

    print(f"Embedding shape: {embeddings.shape}")

    # Get embedding dimension
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    # Save index
    faiss.write_index(index, str(INDEX_FILE))

    print(f"FAISS index created with {index.ntotal} vectors")
    print(f"Saved to: {INDEX_FILE}")


if __name__ == "__main__":
    create_faiss_index()