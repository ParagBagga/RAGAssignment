import requests


OLLAMA_URL = "http://localhost:11434"
MODEL = "llama32-local:latest"


def generate_answer(prompt):

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


if __name__ == "__main__":

    prompt = """
Explain self-attention in simple words.
"""

    answer = generate_answer(prompt)

    print("\nAnswer:\n")
    print(answer)