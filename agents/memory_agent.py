from typing import List, Tuple

class ConversationMemoryAgent:
    def __init__(self, max_memory_length: int = 5):
        self.memory: List[Tuple[str, str]] = []
        self.max_memory_length = max_memory_length

    def add_interaction(self, user_prompt: str, system_response: str):
        self.memory.append((user_prompt, system_response))

        if len(self.memory) > self.max_memory_length:
            self.memory = self.memory[-self.max_memory_length:]

    def get_context(self) -> str:
        context = "Conversation History:\n"
        for user, response in self.memory:
            context += f"User: {user}\nAssistant: {response}\n\n"
        return context
