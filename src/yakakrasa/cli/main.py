import click

@click.group()
def main():
    """
    YakaKrasa CLI tool.
    """
    pass

@main.command()
def train():
    """Train a new model."""
    click.echo("Training a new model...")

@main.command()
def chat():
    """Chat with a trained model."""
    click.echo("Starting a chat session...")

if __name__ == '__main__':
    main()
