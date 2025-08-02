# YakaKrasa Examples

## Quick Start

### 1. Train a model
```bash
yakakrasa train --data examples/sample_data.json --model-path my_model.pt --epochs 100
```

### 2. Test predictions
```bash
yakakrasa predict --model-path my_model.pt "hello there"
yakakrasa predict --model-path my_model.pt "thanks for the help"
```

### 3. Interactive demo
```bash
yakakrasa demo --model-path my_model.pt
```

## Data Format

Training data should be a JSON file with this structure:
```json
[
  {"text": "hello", "intent": "greet"},
  {"text": "goodbye", "intent": "farewell"},
  ...
]
```

Each item needs:
- `text`: The input text to classify
- `intent`: The target intent label