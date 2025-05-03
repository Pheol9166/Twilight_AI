import json


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_config(path="./config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
