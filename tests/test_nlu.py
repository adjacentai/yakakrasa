import pytest

from yakakrasa.core.nlu.pipeline import Pipeline
from yakakrasa.core.nlu.tokenizer import Tokenizer, Token
from yakakrasa.core.nlu.featurizer import Featurizer
from yakakrasa.core.nlu.intent_classifier import IntentClassifier
from yakakrasa.core.nlu.entity_extractor import EntityExtractor
from yakakrasa.core.models.intent_classifier import IntentClassifier as IntentClassifierModel
import numpy as np

def test_tokenizer_process():
    tokenizer = Tokenizer()
    message = {"text": "hello world"}
    tokenizer.process(message)
    tokens = message["tokens"]
    assert len(tokens) == 2
    assert tokens[0].text == "hello"
    assert tokens[1].text == "world"
    assert tokens[0].start == 0
    assert tokens[1].start == 6

def test_pipeline():
    pipeline = Pipeline(components=[Tokenizer()])
    text = "let's build an empire"
    result = pipeline.process(text)
    assert "tokens" in result
    assert len(result["tokens"]) == 4
    assert result["tokens"][0].text == "let's"
    assert result["tokens"][3].text == "empire"

def test_featurizer():
    # 1. Prepare training data
    training_texts = ["hello world", "hello there"]
    tokenizer = Tokenizer()
    training_tokens = []
    for text in training_texts:
        msg = {"text": text}
        tokenizer.process(msg)
        training_tokens.append(msg["tokens"])

    # 2. Fit the featurizer
    featurizer = Featurizer()
    featurizer.fit(training_tokens)

    # Vocab should be {'hello', 'there', 'world'}
    assert featurizer.vocab_size == 3
    assert "hello" in featurizer.vocab

    # 3. Test processing
    pipeline = Pipeline(components=[tokenizer, featurizer])
    test_text = "hello you"
    result = pipeline.process(test_text)

    assert "features" in result
    features = result["features"].toarray()
    assert features.shape == (1, 3)
    # 'hello' is in vocab, 'you' is not
    assert features[0, featurizer.vocab['hello']] == 1.0
    # The vector should have only one non-zero element
    assert np.count_nonzero(features) == 1

def test_intent_classifier():
    # 1. Prepare components and training data
    training_texts = ["hello world", "goodbye world"]
    tokenizer = Tokenizer()
    training_tokens = []
    for text in training_texts:
        msg = {"text": text}
        tokenizer.process(msg)
        training_tokens.append(msg["tokens"])

    featurizer = Featurizer()
    featurizer.fit(training_tokens)

    # 2. Create a dummy model and intent map
    vocab_size = featurizer.vocab_size
    num_intents = 2
    intent_map = {0: "greet", 1: "farewell"}
    
    torch_model = IntentClassifierModel(input_size=vocab_size, hidden_size=10, output_size=num_intents)
    intent_classifier = IntentClassifier(torch_model, intent_map)

    # 3. Build and run the pipeline
    pipeline = Pipeline(components=[tokenizer, featurizer, intent_classifier])
    test_text = "hello you"
    result = pipeline.process(test_text)

    # 4. Check results
    assert "intent" in result
    assert "name" in result["intent"]
    assert "confidence" in result["intent"]
    assert result["intent"]["name"] in ["greet", "farewell"]
    assert result["intent"]["confidence"] > 0 

def test_entity_extractor():
    """Test entity extraction functionality."""
    extractor = EntityExtractor()
    text = "Забронируй столик на 8 вечера завтра"
    message = {"text": text}
    
    extractor.process(message)
    entities = message["entities"]
    
    assert len(entities) >= 2  # Should find time and date
    time_entities = [e for e in entities if e.entity_type == "time"]
    date_entities = [e for e in entities if e.entity_type == "date"]
    
    assert len(time_entities) > 0
    assert len(date_entities) > 0
    assert "8 вечера" in [e.text for e in time_entities]
    assert "завтра" in [e.text for e in date_entities]

def test_pipeline_with_entities():
    """Test full pipeline including entity extraction."""
    # First, prepare some training data to fit the featurizer
    training_texts = ["позвони мне", "напомни завтра"]
    tokenizer = Tokenizer()
    training_tokens = []
    for text in training_texts:
        msg = {"text": text}
        tokenizer.process(msg)
        training_tokens.append(msg["tokens"])
    
    # Create and fit featurizer
    featurizer = Featurizer()
    featurizer.fit(training_tokens)
    
    # Now create the full pipeline
    pipeline = Pipeline(components=[
        Tokenizer(),
        EntityExtractor(),
        featurizer
    ])
    
    text = "Позвони мне в 3 часа дня"
    result = pipeline.process(text)
    
    assert "tokens" in result
    assert "entities" in result
    assert "features" in result
    
    # Check that entities were extracted
    entities = result["entities"]
    number_entities = [e for e in entities if e.entity_type == "number"]
    assert len(number_entities) > 0
    assert "3" in [e.text for e in number_entities] 