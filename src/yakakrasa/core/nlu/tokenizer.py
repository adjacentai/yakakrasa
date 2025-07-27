from typing import List, Dict, Any, Text
from yakakrasa.core.nlu.pipeline import Component

class Token:
    def __init__(self, text: str, start: int):
        self.text = text
        self.start = start
        self.end = start + len(text)

    def __repr__(self):
        return f"Token('{self.text}', {self.start})"

class Tokenizer(Component):
    def process(self, message: Dict[Text, Any]) -> None:
        """
        A simple tokenizer that splits text by space.
        Adds a 'tokens' attribute to the message.
        """
        text = message.get("text", "")
        # A simple regex tokenizer could be an improvement
        words = text.lower().split()
        
        tokens = []
        offset = 0
        for word in words:
            # This find logic is flawed, but good enough for now
            start = text.find(word, offset)
            tokens.append(Token(word, start))
            offset = start + len(word)

        message["tokens"] = tokens
