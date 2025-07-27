import pytest

from yakakrasa.core.nlu.pipeline import Pipeline
from yakakrasa.core.nlu.tokenizer import Tokenizer, Token
from yakakrasa.core.nlu.featurizer import Featurizer
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