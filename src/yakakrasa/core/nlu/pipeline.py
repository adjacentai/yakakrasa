from typing import List, Dict, Any, Text

class Component:
    def process(self, message: Dict[Text, Any]) -> None:
        """
        Process an incoming message.

        This is the method every component in a pipeline
        should implement.
        """
        pass

class Pipeline:
    def __init__(self, components: List[Component]):
        self.components = components

    def process(self, text: Text) -> Dict[Text, Any]:
        message = {"text": text}
        for component in self.components:
            component.process(message)
        return message
