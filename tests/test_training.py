import pytest
import torch
import numpy as np

from yakakrasa.core.nlu.pipeline import Pipeline
from yakakrasa.core.nlu.tokenizer import Tokenizer
from yakakrasa.core.nlu.featurizer import Featurizer
from yakakrasa.core.nlu.intent_classifier import IntentClassifier
from yakakrasa.core.models.intent_classifier import IntentClassifier as IntentClassifierModel
from yakakrasa.core.train.trainer import Trainer

def test_trainer():
    # 1. Prepare data and pipeline
    train_examples = [
        ("hello there", "greet"),
        ("good morning", "greet"),
        ("bye bye now", "farewell"),
        ("see you later", "farewell"),
    ]
    intent_to_id = {"greet": 0, "farewell": 1}
    id_to_intent = {v: k for k, v in intent_to_id.items()}
    num_intents = len(intent_to_id)

    tokenizer = Tokenizer()
    featurizer = Featurizer()

    # Pre-process training data to fit featurizer
    processed_train_data = []
    for text, intent in train_examples:
        msg = {"text": text}
        tokenizer.process(msg)
        processed_train_data.append(msg)
    
    featurizer.fit([msg["tokens"] for msg in processed_train_data])

    # Finalize processing
    for msg in processed_train_data:
        featurizer.process(msg)
        msg["intent_id"] = intent_to_id[next(i[1] for i in train_examples if i[0] == msg["text"])]

    # 2. Create model and pipeline
    vocab_size = featurizer.vocab_size
    model = IntentClassifierModel(vocab_size, 8, num_intents)
    intent_classifier = IntentClassifier(model, id_to_intent)
    pipeline = Pipeline(components=[tokenizer, featurizer, intent_classifier])

    # 3. Check prediction BEFORE training
    test_text = "hello"
    msg_before = pipeline.process(test_text)
    pred_before = msg_before["intent"]["name"]

    # 4. Train the model
    trainer = Trainer(pipeline, processed_train_data)
    trainer.train(epochs=10, batch_size=2, learning_rate=0.1)

    # 5. Check prediction AFTER training
    msg_after = pipeline.process(test_text)
    pred_after = msg_after["intent"]["name"]
    confidence_after = msg_after["intent"]["confidence"]

    assert pred_after == "greet"
    assert confidence_after > 0.5 # Confidence should be higher 