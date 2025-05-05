from dotenv import load_dotenv
from huggingface_hub import login
from app.services.models import load_llm, load_embedding_model
import os
import json


def api_login():
    load_dotenv()
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise Exception("API token not found")
    login(token=api_token)


def model_setting(config):
    embedding_model = load_embedding_model(config)
    llm = load_llm(config)
    return embedding_model, llm


def load_config(path="./config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(path="./prompt.txt"):
    with open(path, "r", encoding="utf-8") as fr:
        return fr.read()
