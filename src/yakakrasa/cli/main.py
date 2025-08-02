import click
import json
import torch
from pathlib import Path

from yakakrasa.core.nlu.pipeline import Pipeline
from yakakrasa.core.nlu.tokenizer import Tokenizer
from yakakrasa.core.nlu.featurizer import Featurizer
from yakakrasa.core.nlu.intent_classifier import IntentClassifier
from yakakrasa.core.models.intent_classifier import IntentClassifier as IntentClassifierModel
from yakakrasa.core.train.trainer import Trainer

@click.group()
def main():
    """
    YakaKrasa CLI tool - Build powerful NLU models with ease.
    """
    pass

@main.command()
@click.option('--data', '-d', required=True, help='Path to training data (JSON file)')
@click.option('--model-path', '-m', default='./model.pt', help='Path to save trained model')
@click.option('--epochs', '-e', default=50, help='Number of training epochs')
@click.option('--lr', default=0.01, help='Learning rate')
def train(data, model_path, epochs, lr):
    """Train a new NLU model on your data."""
    click.echo(f"🚀 Training YakaKrasa model on {data}")
    
    # Load training data
    with open(data, 'r') as f:
        train_examples = json.load(f)
    
    # Extract intents and build mappings
    intents = list(set(example['intent'] for example in train_examples))
    intent_to_id = {intent: i for i, intent in enumerate(intents)}
    id_to_intent = {i: intent for intent, i in intent_to_id.items()}
    
    click.echo(f"📚 Found {len(intents)} intents: {', '.join(intents)}")
    
    # Build pipeline components
    tokenizer = Tokenizer()
    featurizer = Featurizer()
    
    # Process training data
    processed_data = []
    for example in train_examples:
        msg = {"text": example['text']}
        tokenizer.process(msg)
        processed_data.append(msg)
    
    # Fit featurizer
    featurizer.fit([msg["tokens"] for msg in processed_data])
    click.echo(f"🔤 Built vocabulary of {featurizer.vocab_size} words")
    
    # Finalize processing
    for i, msg in enumerate(processed_data):
        featurizer.process(msg)
        msg["intent_id"] = intent_to_id[train_examples[i]['intent']]
    
    # Create model and pipeline
    model = IntentClassifierModel(featurizer.vocab_size, 64, len(intents))
    intent_classifier = IntentClassifier(model, id_to_intent)
    pipeline = Pipeline([tokenizer, featurizer, intent_classifier])
    
    # Train
    trainer = Trainer(pipeline, processed_data)
    click.echo(f"🏋️ Training for {epochs} epochs...")
    trainer.train(epochs=epochs, batch_size=8, learning_rate=lr)
    
    # Save model and metadata
    model_data = {
        'model_state_dict': model.state_dict(),
        'vocab': featurizer.vocab,
        'intent_map': id_to_intent,
        'vocab_size': featurizer.vocab_size,
        'num_intents': len(intents)
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
    
    # Load model
    model_data = torch.load(model_path)
    
    # Recreate pipeline
    tokenizer = Tokenizer()
    featurizer = Featurizer()
    featurizer.vocab = model_data['vocab']
    featurizer.vocab_size = model_data['vocab_size']
    
    model = IntentClassifierModel(
        model_data['vocab_size'], 
        64, 
        model_data['num_intents']
    )
    model.load_state_dict(model_data['model_state_dict'])
    
    intent_classifier = IntentClassifier(model, model_data['intent_map'])
    pipeline = Pipeline([tokenizer, featurizer, intent_classifier])
    
    # Predict
    result = pipeline.process(text)
    intent = result['intent']
    
    click.echo(f"🎯 Text: '{text}'")
    click.echo(f"📊 Intent: {intent['name']} (confidence: {intent['confidence']:.3f})")

@main.command()
@click.option('--model-path', '-m', default='./model.pt', help='Path to trained model')
def demo(model_path):
    """Interactive demo of your trained model."""
    if not Path(model_path).exists():
        click.echo(f"❌ Model not found at {model_path}")
        return
    
    click.echo("🎪 YakaKrasa Interactive Demo")
    click.echo("Type 'quit' to exit\n")
    
    # Load model (same as predict command)
    model_data = torch.load(model_path)
    tokenizer = Tokenizer()
    featurizer = Featurizer()
    featurizer.vocab = model_data['vocab']
    featurizer.vocab_size = model_data['vocab_size']
    
    model = IntentClassifierModel(
        model_data['vocab_size'], 
        64, 
        model_data['num_intents']
    )
    model.load_state_dict(model_data['model_state_dict'])
    
    intent_classifier = IntentClassifier(model, model_data['intent_map'])
    pipeline = Pipeline([tokenizer, featurizer, intent_classifier])
    
    while True:
        try:
            text = click.prompt("Enter text", type=str)
            if text.lower() == 'quit':
                break
            
            result = pipeline.process(text)
            intent = result['intent']
            click.echo(f"➡️  Intent: {intent['name']} (confidence: {intent['confidence']:.3f})\n")
            
        except (KeyboardInterrupt, EOFError):
            break
    
    click.echo("👋 Goodbye!")

if __name__ == '__main__':
    main()