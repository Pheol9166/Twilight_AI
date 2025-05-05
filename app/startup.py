from app.utils.utils import load_config, load_prompt, model_setting, api_login


def initialize_app(config_path: str= "./config.json", prompt_path: str= "./prompt.txt"):
    api_login()
    config = load_config(config_path)
    prompt = load_prompt(prompt_path)
    embedding_model, llm = model_setting(config)
    return config, prompt, embedding_model, llm
