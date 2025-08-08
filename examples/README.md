# YakaKrasa Examples

## Quick Start

### 1. Train a model
```bash
yakakrasa train --config examples/config.yml --data examples/sample_data.json --model-path my_model.pt
```

### 2. Test predictions
```bash
yakakrasa predict --model-path my_model.pt "hello there"
yakakrasa predict --model-path my_model.pt "thanks for the help"
```

### 3. Evaluate model
```bash
yakakrasa evaluate --model-path my_model.pt --data examples/sample_eval.json
```

### 4. Interactive demo
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

Evaluation data can also contain entities:
```json
[
  {"text": "call me at 8 pm", "intent": "greet", "entities": [{"text": "8 pm", "entity": "time"}]}
]
```

Each item needs:
- `text`: The input text to classify
- `intent`: The target intent label
- `entities` (optional): list of `{text, entity}` objects