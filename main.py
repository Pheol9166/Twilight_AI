from fastapi import FastAPI
from app.startup import initialize_app
from app.routes import build_routes

app = FastAPI()

config, prompt, embedding_model, llm = initialize_app(
    config_path="app/config/config.json", prompt_path="app/config/prompt.txt"
)

build_routes(app, config, prompt, embedding_model, llm)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
