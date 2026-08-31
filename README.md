# RAG Assignment

A Retrieval-Augmented Generation (RAG) application that answers questions based on PDF documents.

The project uses local AI models through Ollama for answer generation, embeddings, and RAG evaluation.

## Technologies Used

- Python
- Ollama
- Llama 3.2
- Qwen 3B
- nomic-embed-text
- LangChain
- FAISS
- RAGAS
- PyPDF
- Pandas



# Architecture


PDF Documents
      |
      v
PDF Loader
      |
      v
Text Chunking
      |
      v
Generate Embeddings
      |
      v
FAISS Vector Store
      |
      v
User Question
      |
      v
Vector Similarity Search
      |
      v
Retrieve Relevant Context
      |
      v
LLM
      |
      v
Generated Answer


## Project Structure

RAGAssignment/
│
├── data/
│   ├── raw/
│   │
│   ├── processed/
│   │   ├── chunks.json
│   │   ├── embedded_chunks.json
│   │   └── faiss_index.bin
│   │
│   ├── evaluation_results.json
│   └── ragas_evaluation_results.csv
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   └── embeddings.py
│   │
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   └── search.py
│   │
│   ├── generation/
│   │   └── llm.py
│   │
│   ├── memory/
│   │   └── conversation_memory.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── ragas_evaluation.py
│   │
│   └── rag.py
│
├── requirements.txt
└── README.md


## Prerequisites

Before running the project, install:

Python 3.10 or above
Ollama
Install Ollama

Ollama was deployed inside a Docker container, and all required local models (Llama 3.2, Qwen 2.5 3B, and nomic-embed-text) were pulled and managed within the Ollama environment. The Python RAG application communicated with the Ollama service through http://localhost:11434.



Ollama Models

This project uses three local models.

Llama 3.2

Used for generating answers from the retrieved document context.

# Install:


ollama pull 
llama3.2
Qwen 3B   (Used for local RAGAS evaluation.)

# Install:

ollama pull qwen2.5:3b
Nomic Embed Text

# Install:

ollama pull
nomic-embed-text

Installation
1. Clone the Repository
git clone <https://github.com/ParagBagga/RAGAssignment.git>
cd RAGAssignment
2. Create a Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / WSL
python -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

4. Place PDF documents inside the following directory:

data/pdfs/

These documents will be used as the knowledge base for the RAG application.

Step 1: Load and Chunk PDFs

Run:

python -m src.ingestion.pdf_loader

This process:

Reads PDF documents
Extracts text from each page
Splits the text into smaller chunks
Saves the chunks

The chunks are saved to:

data/processed/chunks.json

The project uses recursive text splitting:

RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
Step 2: Generate Embeddings

Run:

python -m src.ingestion.embeddings

This process:

Loads the generated text chunks
Sends each chunk to the Ollama embedding model
Generates vector embeddings
Saves the embedded chunks

The output file is:

data/processed/embedded_chunks.json

The embedding model used is:

nomic-embed-text
Step 3: Create FAISS Vector Index

Run:

python -m src.retrieval.vector_store

This process:

Loads embedded chunks
Extracts embedding vectors
Creates a FAISS vector index
Saves the vector index

The FAISS index is saved to:

data/processed/faiss_index.bin
Step 4: Run the RAG Application

Run:

python -m src.rag

Example:

RAG Assistant Started!

Ask a question:

Example question:

What is the Transformer architecture?

The application performs the following process:

User Question
      |
      v
Create Query Embedding
      |
      v
FAISS Similarity Search
      |
      v
Retrieve Relevant Chunks
      |
      v
Build Context
      |
      v
Send Context + Question to LLM
      |
      v
Generate Answer

The application retrieves the top relevant chunks using:

search(question, top_k=4)
Conversation Memory

The application maintains recent conversation history.

The memory configuration is:

memory = ConversationMemory(max_turns=4)

This allows the application to understand follow-up questions and references to previous questions.

Answer Generation

Answer generation is handled in:

src/generation/llm.py

The application sends prompts to the local Ollama API.

Example configuration:

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2"

The RAG prompt instructs the model to answer document-related questions using only the retrieved document context.

If the answer is unavailable in the documents, the application responds:

I don't have enough information in the provided documents.
Evaluation

The project uses RAGAS to evaluate the RAG pipeline.

The evaluation process measures:

Faithfulness
Answer Correctness
Context Recall
Context Precision
Generate Evaluation Data

Run:

python -m src.evaluation.evaluate

This process:

Loads evaluation questions
Runs the questions through the RAG pipeline
Stores generated answers
Stores retrieved contexts
Saves the evaluation data

The output file is:

data/evaluation_results.json
Run RAGAS Evaluation

Run:

python -m src.evaluation.ragas_evaluation

The evaluation uses:

Qwen 3B as the local evaluator LLM
nomic-embed-text for embeddings

The results are saved to:

data/ragas_evaluation_results.csv
RAGAS Metrics
Faithfulness

Checks whether the generated answer is supported by the retrieved context.

A higher score means the answer is less likely to contain hallucinated information.

Answer Correctness

Checks how correct the generated answer is compared with the reference answer.

Context Recall

Checks whether the retrieved context contains enough information required to answer the question.

Context Precision

Checks whether the retrieved chunks are relevant to the question.

Example Evaluation Results

Example output:

Total Questions Evaluated: 2

Overall Scores:

Faithfulness: 100.00%
Answer Correctness: 53.64%
Context Recall: 66.67%
Context Precision: 100.00%

EVALUATION COMPLETE
Improving RAG Performance

The following parameters can be adjusted to improve the RAG system.

Chunk Size

Current configuration:

chunk_size=1000

Smaller chunks may improve retrieval precision.

Larger chunks may provide more context.

Chunk Overlap

Current configuration:

chunk_overlap=200

Chunk overlap helps preserve information between adjacent chunks.

Number of Retrieved Chunks

Current configuration:

top_k=4

Increasing top_k may improve context recall but can also introduce irrelevant information.

Full Pipeline

To rebuild the entire RAG system from scratch, run the following commands:

# Step 1: Process PDF documents
python -m src.ingestion.pdf_loader

# Step 2: Generate embeddings
python -m src.ingestion.embeddings

# Step 3: Create FAISS vector index
python -m src.retrieval.vector_store

# Step 4: Run the RAG application
python -m src.rag

To run the evaluation:

# Step 5: Generate evaluation data
python -m src.evaluation.evaluate

# Step 6 :Run RAGAS evaluation
python -m src.evaluation.ragas_evaluation
Requirements

Install all Python dependencies using:

pip install -r requirements.txt

The project requires:

langchain
langchain-community
langchain-ollama
ragas
datasets
pandas
faiss-cpu
numpy
pypdf
requests
python-dotenv

Note: Ollama models are not installed using pip.

Install them separately:

ollama pull llama3.2
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
Future Improvements

Possible improvements include:

Hybrid search using keyword and vector search
Document reranking
Improved chunking strategies
Metadata filtering
Streaming responses
Web interface using Streamlit
Persistent conversation memory
Larger evaluation datasets
Improved prompt engineering
Testing different embedding models
Testing different local LLMs
Summary

This project demonstrates a complete local Retrieval-Augmented Generation pipeline.

PDF Documents
      |
      v
Text Extraction
      |
      v
Chunking
      |
      v
Embeddings
      |
      v
FAISS Vector Store
      |
      v
Similarity Search
      |
      v
Context Retrieval
      |
      v
LLM Answer Generation
      |
      v
RAGAS Evaluation

The project demonstrates the complete lifecycle of building, running, and evaluating a RAG application using local models.

