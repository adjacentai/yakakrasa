import pytest
import json
import tempfile
import os
from pathlib import Path
from click.testing import CliRunner

from yakakrasa.cli.main import main

# Sample config for testing
CONFIG_YAML = """
language: "en"
pipeline:
  - name: "Tokenizer"
  - name: "Featurizer"
  - name: "IntentClassifier"
    model:
      hidden_size: 10
    trainer:
      epochs: 5
      batch_size: 2
      learning_rate: 0.1
"""

def test_cli_train_and_predict_with_config():
    """Test the full CLI workflow using a config file."""
    runner = CliRunner()
    
    # Create temporary training data
    train_data = [
        {"text": "hello", "intent": "greet"},
        {"text": "hi there", "intent": "greet"},
        {"text": "goodbye", "intent": "farewell"},
        {"text": "bye", "intent": "farewell"},
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write training data and config
        data_file = Path(temp_dir) / "train.json"
        config_file = Path(temp_dir) / "config.yml"
        
        with open(data_file, 'w') as f:
            json.dump(train_data, f)
        with open(config_file, 'w') as f:
            f.write(CONFIG_YAML)
            
        # Model path
        model_file = Path(temp_dir) / "model.pt"
        
        # Test training
        result = runner.invoke(main, [
            'train', 
            '--config', str(config_file),
            '--data', str(data_file),
            '--model-path', str(model_file)
        ])
        
        assert result.exit_code == 0, result.output
        assert "Training YakaKrasa model with config" in result.output
        assert "Model saved" in result.output
        assert model_file.exists()
        
        # Test prediction
        result = runner.invoke(main, [
            'predict',
            '--model-path', str(model_file),
            'hello world'
        ])
        
        assert result.exit_code == 0, result.output
        assert "Intent: greet" in result.output
        assert "confidence:" in result.output

def test_cli_predict_missing_model():
    """Test predict command with missing model file."""
    runner = CliRunner()
    
    result = runner.invoke(main, [
        'predict',
        '--model-path', 'nonexistent_model.pt',
        'hello'
    ])
    
    assert result.exit_code == 0
    assert "Model not found" in result.output
