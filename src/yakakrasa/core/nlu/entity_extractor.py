from typing import Dict, Any, Text, List
import re

from yakakrasa.core.nlu.pipeline import Component
from yakakrasa.core.nlu.tokenizer import Token

class Entity:
    def __init__(self, text: str, entity_type: str, start: int, end: int):
        self.text = text
        self.entity_type = entity_type
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Entity('{self.text}', '{self.entity_type}', {self.start}, {self.end})"

class EntityExtractor(Component):
    """
    A simple rule-based entity extractor.
    """
    def __init__(self, entity_patterns: Dict[str, List[str]] = None):
        self.entity_patterns = entity_patterns or {
            'time': [
                r'\b\d{1,2}:\d{2}\b',  # 8:30, 14:45
                r'\b\d{1,2}\s*(?:am|pm|вечера|утра|дня)\b',  # 8 pm, 2 вечера
                r'\b(?:утром|днем|вечером|ночью)\b'
            ],
            'date': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 12/25/2023
                r'\b(?:сегодня|завтра|вчера|позавчера)\b'
            ],
            'number': [
                r'\b\d+\b'  # any number
            ]
        }
        self.compiled_patterns = {
            entity_type: [re.compile(pattern, re.IGNORECASE) 
                         for pattern in patterns]
            for entity_type, patterns in self.entity_patterns.items()
        }

    def process(self, message: Dict[Text, Any]) -> None:
        """
        Extract entities from text and add them to the message.
        """
        text = message.get("text", "")
        entities = []
        
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end()
                    )
                    entities.append(entity)
        
        # Sort entities by start position
        entities.sort(key=lambda x: x.start)
        message["entities"] = entities 