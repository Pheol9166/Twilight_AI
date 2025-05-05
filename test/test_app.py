from fastapi import FastAPI
from app.startup import initialize_app
from app.routes import build_routes

app = FastAPI()

config, prompt, embedding_model, llm = initialize_app(
    config_path="./test/test_config.json",
    prompt_path="./test/test_prompt.txt"
)

build_routes(app, config, prompt, embedding_model, llm)