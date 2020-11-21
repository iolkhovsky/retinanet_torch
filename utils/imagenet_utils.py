import json


def encode_id(idx, config_path):
    with open(config_path) as f:
        labels = json.load(f)
    return labels[idx]
