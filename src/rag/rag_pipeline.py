from src.retrieval.search import search
from src.generation.llm import generate_answer
from src.memory.conversation_memory import ConversationMemory



memory = ConversationMemory(max_turns=4)

def clear_conversation_memory():
    memory.clear()  


def build_context(results):

    context = ""

    for i, result in enumerate(results, start=1):

        context += f"""
--- Context {i} ---
Source: {result['source']}
Page: {result['page']}

{result['text']}
"""

    return context

  


def ask_question(question):

    # Step 1: Retrieve relevant chunks
    results = search(question, top_k=4)

    # Step 2: Build document context
    context = build_context(results)

    # Step 3: Get previous conversation history
    conversation_history = memory.get_history()

    # Step 4: Create prompt
    prompt = f"""
You are a helpful AI assistant.

Use the conversation history to understand previous questions, follow-up questions,
references, and questions about the conversation itself.

If the user asks about something from the conversation history, answer using the
conversation history.

For questions about the documents, answer using ONLY the provided document context.

If the answer is not available in either the conversation history or the provided
documents, say:
"I don't have enough information in the provided documents."
CONVERSATION HISTORY:
{conversation_history}

DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{question}

ANSWER:
"""

    # Step 5: Generate answer
    answer = generate_answer(prompt)

    # Step 6: Save conversation in memory
    memory.add_turn(question, answer)

    return answer, results


if __name__ == "__main__":

    print("RAG Assistant Started!")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask a question: ").strip()

        # Exit BEFORE calling ask_question
        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        answer, sources = ask_question(question)

        print("\nAnswer:")
        print(answer)

        if sources:
            print("\nSources:")

            for source in sources:
                print(
                    f"- {source['source']} "
                    f"(Page {source['page']})"
                )

        print("\n" + "=" * 60 + "\n")