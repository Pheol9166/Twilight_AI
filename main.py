from fastapi import FastAPI
from startup import initialize_app
from routes import build_routes

app = FastAPI()

config, prompt, embedding_model, llm = initialize_app(
    config_path="./config.json",
    prompt_path="./prompt.txt"
)

build_routes(app, config, prompt, embedding_model, llm)