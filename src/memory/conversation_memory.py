class ConversationMemory:

    def __init__(self, max_turns=4):
        self.max_turns = max_turns
        self.history = []

    def add_turn(self, question, answer):

        self.history.append({
            "question": question,
            "answer": answer
        })

        # Keep only last N turns
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_history(self):

        conversation = ""

        for turn in self.history:
            conversation += f"""
User: {turn['question']}

Assistant: {turn['answer']}

"""

        return conversation

    def clear(self):
        self.history = []