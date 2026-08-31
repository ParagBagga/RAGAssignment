import json
import time

from src.rag.rag_pipeline import (
    ask_question,
    clear_conversation_memory
)


def load_questions():

    with open("data/test_questions.json", "r") as file:
        return json.load(file)


def run_evaluation():

    questions = load_questions()

    results = []

    print(f"\nStarting evaluation for {len(questions)} questions...\n")

    for item in questions:

        # Clear memory so each evaluation is independent
        clear_conversation_memory()

        question_id = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"Running Question {question_id}: {question}")

        start_time = time.time()

        # Run question through RAG pipeline
        answer, sources = ask_question(question)

        end_time = time.time()

        response_time = round(end_time - start_time, 2)

        # Extract retrieved contexts
        contexts = [
            source["text"]
            for source in sources
        ]

        result = {
            "id": question_id,
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "response_time_seconds": response_time,
            "sources": [
                {
                    "source": source["source"],
                    "page": source["page"]
                }
                for source in sources
            ]
        }

        results.append(result)

        print(f"Completed in {response_time} seconds\n")

    # Save results
    with open("data/evaluation_results.json", "w") as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("Evaluation completed!")
    print("Results saved to: data/evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()