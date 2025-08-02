import yaml
from typing import Dict, Any, Text, List

from yakakrasa.core.nlu.pipeline import Pipeline
from yakakrasa.core.nlu.tokenizer import Tokenizer
from yakakrasa.core.nlu.featurizer import Featurizer
from yakakrasa.core.nlu.intent_classifier import IntentClassifier
from yakakrasa.core.models.intent_classifier import IntentClassifier as IntentClassifierModel

# Registry of all available components
component_classes = {
    "Tokenizer": Tokenizer,
    "Featurizer": Featurizer,
    "IntentClassifier": IntentClassifier,
}

def load_config(config_path: Text) -> Dict[Text, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_pipeline_from_config(config: Dict[Text, Any], **kwargs) -> Pipeline:
    """Create an NLU pipeline from a configuration dictionary."""
    pipeline_config = config.get("pipeline", [])
    components = []
    
    for component_config in pipeline_config:
        component_name = component_config.get("name")
        if not component_name or component_name not in component_classes:
            raise ValueError(f"Unknown component: {component_name}")
            
        component_class = component_classes[component_name]
        
        # This is a simplified way to pass component-specific dependencies
        # In a real framework, this would be more robust
        if component_name == "IntentClassifier":
            # Pass model and intent_map from kwargs
            model = kwargs.get('model')
            intent_map = kwargs.get('intent_map')
            if not model or not intent_map:
                raise ValueError("IntentClassifier requires a 'model' and 'intent_map'.")
            
            components.append(component_class(model=model, intent_map=intent_map))
        else:
            components.append(component_class())
            
    return Pipeline(components)

