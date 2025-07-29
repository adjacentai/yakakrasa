import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any, Text

from yakakrasa.core.nlu.pipeline import Pipeline
from yakakrasa.core.models.intent_classifier import IntentClassifier

class Trainer:
    def __init__(self, pipeline: Pipeline, train_data: List[Dict[Text, Any]]):
        self.pipeline = pipeline
        self.train_data_raw = train_data
        self.model = self._find_model()

    def _find_model(self) -> IntentClassifier:
        """Finds the trainable model in the pipeline."""
        for component in self.pipeline.components:
            if hasattr(component, 'model') and isinstance(component.model, nn.Module):
                return component.model
        raise ValueError("No trainable model found in the pipeline.")

    def train(self, epochs: int, batch_size: int, learning_rate: float):
        """
        The main training loop.
        """
        # 1. Prepare data
        # In a real framework, this would be much more robust.
        feature_tensors = [torch.from_numpy(d["features"].toarray()) for d in self.train_data_raw]
        X = torch.cat(feature_tensors, dim=0)
        y = torch.tensor([d["intent_id"] for d in self.train_data_raw], dtype=torch.long)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 2. Prepare optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # 3. Training loop
        self.model.train() # Set model to training mode
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                # Forward pass
                y_pred = self.model(X_batch)
                
                # Compute loss
                loss = loss_fn(y_pred, y_batch)
                total_loss += loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        print("Training finished.")
