import os
import json

class ConversationMemory:
    """
    Manages multi-turn conversation memory history for the Gemini Agent.
    Maintains a list of messages formatted for the Google Generative AI API.
    """
    def __init__(self, system_instruction: str = None):
        self.system_instruction = system_instruction
        self.history = []

    def add_user_message(self, text: str):
        """Adds a standard text message from the user to the history."""
        self.history.append({
            "role": "user",
            "parts": [text]
        })

    def add_model_response(self, text: str):
        """Adds a standard text response from the model to the history."""
        self.history.append({
            "role": "model",
            "parts": [text]
        })

    def add_raw_message(self, message: dict):
        """
        Adds a raw message dictionary directly. 
        Useful for complex objects like tool calls and tool responses.
        Format: {"role": "user"|"model", "parts": [...]}
        """
        self.history.append(message)

    def get_history(self) -> list:
        """Returns the conversation history list."""
        return self.history

    def clear(self):
        """Resets the conversation history."""
        self.history = []

    def to_json(self) -> str:
        """Serializes the conversation history to a JSON string."""
        return json.dumps(self.history, indent=2)

    def load_from_json(self, json_str: str):
        """Loads conversation history from a JSON string."""
        self.history = json.loads(json_str)
