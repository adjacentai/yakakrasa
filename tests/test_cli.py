import pytest
import json
import tempfile
import os
from pathlib import Path
from click.testing import CliRunner

from yakakrasa.cli.main import main

def test_cli_train_and_predict():
    """Test the full CLI workflow: train a model, then predict with it."""
    runner = CliRunner()
    
    # Create temporary training data
    train_data = [
        {"text": "hello", "intent": "greet"},
        {"text": "hi there", "intent": "greet"},
        {"text": "goodbye", "intent": "farewell"},
        {"text": "bye", "intent": "farewell"},
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write training data
        data_file = Path(temp_dir) / "train.json"
        with open(data_file, 'w') as f:
            json.dump(train_data, f)
        
        # Model path
        model_file = Path(temp_dir) / "model.pt"
        
        # Test training
        result = runner.invoke(main, [
            'train', 
            '--data', str(data_file),
            '--model-path', str(model_file),
            '--epochs', '5'  # Quick training for test
        ])
        
        assert result.exit_code == 0
        assert "Training YakaKrasa model" in result.output
        assert "Model saved" in result.output
        assert model_file.exists()
        
        # Test prediction
        result = runner.invoke(main, [
            'predict',
            '--model-path', str(model_file),
            'hello world'
        ])
        
        assert result.exit_code == 0
        assert "Intent:" in result.output
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