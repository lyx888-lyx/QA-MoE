import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)
