import click
import json
import torch
from pathlib import Path

from yakakrasa.core.config import load_config, create_pipeline_from_config
from yakakrasa.core.train.trainer import Trainer
from yakakrasa.core.models.intent_classifier import IntentClassifier as IntentClassifierModel

@click.group()
def main():
    """
    YakaKrasa CLI tool - Build powerful NLU models with ease.
    """
    pass

@main.command()
@click.option('--config', '-c', required=True, help='Path to configuration file')
@click.option('--data', '-d', required=True, help='Path to training data (JSON file)')
@click.option('--model-path', '-m', default='./model.pt', help='Path to save trained model')
def train(config, data, model_path):
    """Train a new NLU model using a configuration file."""
    click.echo(f"🚀 Training YakaKrasa model with config {config}")
    
    # Load config and data
    config_data = load_config(config)
    with open(data, 'r') as f:
        train_examples = json.load(f)
    
    # Extract intents and build mappings
    intents = list(set(example['intent'] for example in train_examples))
    intent_to_id = {intent: i for i, intent in enumerate(intents)}
    id_to_intent = {i: intent for intent, i in intent_to_id.items()}
    
    click.echo(f"📚 Found {len(intents)} intents: {', '.join(intents)}")
    
    # Prepare a dummy pipeline for data processing
    tokenizer = create_pipeline_from_config({"pipeline": [{"name": "Tokenizer"}]}).components[0]
    featurizer = create_pipeline_from_config({"pipeline": [{"name": "Featurizer"}]}).components[0]
    
    processed_data = []
    for example in train_examples:
        msg = {"text": example['text']}
        tokenizer.process(msg)
        processed_data.append(msg)
    
    # Fit featurizer
    featurizer.fit([msg["tokens"] for msg in processed_data])
    click.echo(f"🔤 Built vocabulary of {featurizer.vocab_size} words")
    
    for i, msg in enumerate(processed_data):
        featurizer.process(msg)
        msg["intent_id"] = intent_to_id[train_examples[i]['intent']]
    
    # Create the real model and pipeline from config
    intent_classifier_config = next(c for c in config_data['pipeline'] if c['name'] == 'IntentClassifier')
    model = IntentClassifierModel(
        input_size=featurizer.vocab_size,
        hidden_size=intent_classifier_config['model']['hidden_size'],
        output_size=len(intents)
    )
    pipeline = create_pipeline_from_config(config_data, model=model, intent_map=id_to_intent)
    
    # Train
    trainer_config = intent_classifier_config['trainer']
    trainer = Trainer(pipeline, processed_data)
    click.echo(f"🏋️ Training for {trainer_config['epochs']} epochs...")
    trainer.train(
        epochs=trainer_config['epochs'],
        batch_size=trainer_config['batch_size'],
        learning_rate=trainer_config['learning_rate']
    )
    
    # Save model and metadata
    model_data = {
        'model_state_dict': model.state_dict(),
        'config': config_data,
        'vocab': featurizer.vocab,
        'intent_map': id_to_intent
    }
    torch.save(model_data, model_path)
    click.echo(f"💾 Model saved to {model_path}")

@main.command()
@click.option('--model-path', '-m', default='./model.pt', help='Path to trained model')
@click.argument('text')
def predict(model_path, text):
    """Predict intent for given text."""
    if not Path(model_path).exists():
        click.echo(f"❌ Model not found at {model_path}")
        return
    
    # Load model and config
    model_data = torch.load(model_path)
    config = model_data['config']
    
    # Recreate model
    intent_classifier_config = next(c for c in config['pipeline'] if c['name'] == 'IntentClassifier')
    model = IntentClassifierModel(
        input_size=len(model_data['vocab']),
        hidden_size=intent_classifier_config['model']['hidden_size'],
        output_size=len(model_data['intent_map'])
    )
    model.load_state_dict(model_data['model_state_dict'])
    
    # Recreate pipeline
    pipeline = create_pipeline_from_config(
        config,
        model=model,
        intent_map=model_data['intent_map']
    )
    # Manually set featurizer vocab
    for component in pipeline.components:
        if hasattr(component, 'vocab'):
            component.vocab = model_data['vocab']
            component.vocab_size = len(model_data['vocab'])

    # Predict
    result = pipeline.process(text)
    intent = result['intent']
    
    click.echo(f"🎯 Text: '{text}'")
    click.echo(f"📊 Intent: {intent['name']} (confidence: {intent['confidence']:.3f})")
    
    # Show entities if any were extracted
    if 'entities' in result and result['entities']:
        click.echo("🏷️  Entities:")
        for entity in result['entities']:
            click.echo(f"   • {entity.entity_type}: '{entity.text}' (pos: {entity.start}-{entity.end})")
    else:
        click.echo("🏷️  No entities found")

@main.command()
@click.option('--model-path', '-m', default='./model.pt', help='Path to trained model')
def demo(model_path):
    """Interactive demo of your trained model."""
    # (This can be updated similarly to `predict`)
    if not Path(model_path).exists():
        click.echo(f"❌ Model not found at {model_path}. Please train a model first with `yakakrasa train`.")
        return

    click.echo("🎪 YakaKrasa Interactive Demo")
    click.echo("Type 'quit' to exit\n")

    model_data = torch.load(model_path)
    config = model_data['config']
    
    intent_classifier_config = next(c for c in config['pipeline'] if c['name'] == 'IntentClassifier')
    model = IntentClassifierModel(
        input_size=len(model_data['vocab']),
        hidden_size=intent_classifier_config['model']['hidden_size'],
        output_size=len(model_data['intent_map'])
    )
    model.load_state_dict(model_data['model_state_dict'])
    
    pipeline = create_pipeline_from_config(config, model=model, intent_map=model_data['intent_map'])
    for component in pipeline.components:
        if hasattr(component, 'vocab'):
            component.vocab = model_data['vocab']
            component.vocab_size = len(model_data['vocab'])
            
    while True:
        try:
            text = click.prompt("Enter text", type=str)
            if text.lower() == 'quit':
                break
            
            result = pipeline.process(text)
            intent = result['intent']
            click.echo(f"➡️  Intent: {intent['name']} (confidence: {intent['confidence']:.3f})")
            
            # Show entities
            if 'entities' in result and result['entities']:
                click.echo("🏷️  Entities:")
                for entity in result['entities']:
                    click.echo(f"   • {entity.entity_type}: '{entity.text}'")
            else:
                click.echo("🏷️  No entities found")
            click.echo()
            
        except (KeyboardInterrupt, EOFError):
            break
    
    click.echo("👋 Goodbye!")

if __name__ == '__main__':
    main()