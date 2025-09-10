from typing import List, Dict, Any, Text, Tuple

def validate_training_data(data: List[Dict[Text, Any]]) -> Tuple[bool, List[str]]:
    """Validate training data format and content."""
    errors = []
    
    if not data:
        errors.append("Training data is empty")
        return False, errors
    
    if not isinstance(data, list):
        errors.append("Training data must be a list")
        return False, errors
    
    for i, example in enumerate(data):
        if not isinstance(example, dict):
            errors.append(f"Example {i} is not a dictionary")
            continue
            
        if 'text' not in example or not isinstance(example['text'], str) or not example['text'].strip():
            errors.append(f"Example {i} has invalid 'text' field")
            
        if 'intent' not in example or not isinstance(example['intent'], str) or not example['intent'].strip():
            errors.append(f"Example {i} has invalid 'intent' field")
    
    if len(data) < 2:
        errors.append("Need at least 2 training examples")
    
    intents = [ex.get('intent') for ex in data if isinstance(ex, dict) and 'intent' in ex]
    if len(set(intents)) < 2:
        errors.append("Need at least 2 different intents")
    
    return len(errors) == 0, errors
