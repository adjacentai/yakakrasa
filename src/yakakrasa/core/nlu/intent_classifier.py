from typing import Dict, Any, Text
import torch
import numpy as np

from yakakrasa.core.nlu.pipeline import Component
from yakakrasa.core.models.intent_classifier import IntentClassifier as IntentClassifierModel

class IntentClassifier(Component):
    """
    An NLU component that uses a PyTorch model to classify intents.
    """
    def __init__(self, model: IntentClassifierModel, intent_map: Dict[int, Text]):
        self.model = model
        self.intent_map = intent_map

    def process(self, message: Dict[Text, Any]) -> None:
        """
        Predict intent for a message and add it to the message object.
        """
        if "features" not in message:
            raise ValueError("No features found in message. Did you run a Featurizer?")

        features = message["features"]
        
        # PyTorch model expects a tensor
        tensor_in = torch.from_numpy(features.toarray())

        # Get model prediction
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            prediction = self.model(tensor_in)
        
        predicted_idx = torch.argmax(prediction).item()
        confidence = torch.max(prediction).item()

        message["intent"] = {
            "name": self.intent_map.get(predicted_idx, "unknown"),
            "confidence": float(confidence)
        } 