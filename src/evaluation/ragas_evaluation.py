import json
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextRecall,
    ContextPrecision,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama, OllamaEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

# Local Ollama evaluator model
OLLAMA_MODEL = "qwen2.5:3b"

# Local embedding model
EMBEDDING_MODEL = "nomic-embed-text"

# Ollama URL
OLLAMA_URL = "http://localhost:11434"

# Evaluate ONLY first 2 questions
NUM_QUESTIONS = 2


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

def load_evaluation_data():

    with open("data/evaluation_results.json", "r") as file:
        results = json.load(file)

    # Take only first 2 questions
    results = results[:NUM_QUESTIONS]

    data = {
        "question": [],
        "answer": [],
        "ground_truth": [],
        "contexts": []
    }

    for item in results:

        data["question"].append(item["question"])
        data["answer"].append(item["answer"])
        data["ground_truth"].append(item["ground_truth"])
        data["contexts"].append(item["contexts"])

    return Dataset.from_dict(data)


# ============================================================
# RUN RAGAS EVALUATION
# ============================================================

def run_ragas_evaluation():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading evaluation data...")

    dataset = load_evaluation_data()

    print(f"Loaded {len(dataset)} evaluation records")
    print(f"Evaluating only first {NUM_QUESTIONS} questions")


    # --------------------------------------------------------
    # LOCAL OLLAMA LLM
    # --------------------------------------------------------

    print("\nConnecting to local Ollama evaluator...")

    langchain_llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        num_predict=4096,
    )

    evaluator_llm = LangchainLLMWrapper(
        langchain_llm
    )


    # --------------------------------------------------------
    # LOCAL OLLAMA EMBEDDINGS
    # --------------------------------------------------------

    print("Connecting to local Ollama embeddings...")

    langchain_embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        langchain_embeddings
    )


    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    print("\nConfiguration:")
    print(f"Evaluator LLM: {OLLAMA_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")


    # --------------------------------------------------------
    # RAGAS METRICS
    # --------------------------------------------------------

    print("\nConfiguring RAGAS metrics...")

    metrics = [

        Faithfulness(
            llm=evaluator_llm
        ),

        AnswerCorrectness(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        ),

        ContextRecall(
            llm=evaluator_llm
        ),

        ContextPrecision(
            llm=evaluator_llm
        )

    ]


    # --------------------------------------------------------
    # RAGAS RUN CONFIGURATION
    # --------------------------------------------------------

    run_config = RunConfig(
        timeout=600,
        max_retries=1,
        max_wait=60,
        max_workers=1
    )


    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    print("\nRunning RAGAS evaluation...")
    print(f"Testing {len(dataset)} questions with 4 metrics.")
    print("Using local Ollama for LLM evaluation.")
    print("Using local Ollama for embeddings.\n")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        batch_size=1,
        raise_exceptions=False
    )


    # ========================================================
    # CONVERT RESULTS
    # ========================================================

    print("\nConverting results...")

    results_df = result.to_pandas()


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = "data/ragas_local_2_questions.csv"

    results_df.to_csv(
        output_file,
        index=False
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("LOCAL OLLAMA RAGAS EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal Questions Evaluated: {len(results_df)}")

    metric_names = [
        "faithfulness",
        "answer_correctness",
        "context_recall",
        "context_precision"
    ]

    print("\nOverall Scores:\n")

    for metric in metric_names:

        if metric not in results_df.columns:

            print(
                f"{metric.replace('_', ' ').title()}: Not available"
            )

            continue

        values = pd.to_numeric(
            results_df[metric],
            errors="coerce"
        ).dropna()

        successful = len(values)
        total = len(results_df)

        if successful > 0:

            average = values.mean() * 100

            print(
                f"{metric.replace('_', ' ').title()}: "
                f"{average:.2f}% "
                f"({successful}/{total} successful)"
            )

        else:

            print(
                f"{metric.replace('_', ' ').title()}: "
                f"No valid results "
                f"(0/{total} successful)"
            )


    print("\nDetailed results saved to:")
    print(output_file)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_ragas_evaluation()