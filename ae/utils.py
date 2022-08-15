import argparse

def _setup_parser():
    parser = argparse.ArgumentParser(description="Aprendendo a fazer um autoencoder simples")

    # Show linear transformation using nn.Linear
    parser.add_argument('--lintrans', action='store_true', help='Runs the linear transformation snippet')

    # Import MNIST dataset
    parser.add_argument('--mnist', action='store_true', help='Imports the MNIST dataset')

    # Create construction-test split
    parser.add_argument('--split', action='store_true', help='Splits dataset into train-validation-test subsets')

    # Normalize construction and test sets
    parser.add_argument('--prep', action='store_true', help='Preprocess dataset')

    # Create a simple autoencoder and run it without training
    parser.add_argument('--simpleae', action='store_true', help='Running simple autoencoder (without training)')

    return parser.parse_args()

