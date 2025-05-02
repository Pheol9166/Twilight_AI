import os
from langchain.llms.base import LLM
from huggingface_hub import InferenceClient
from pydantic import Field


class HFChatCompletionLLM(LLM):
    model: str
    api_key: str
    provider: str
    temperature: float
    max_tokens: int
    client: InferenceClient = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.client = InferenceClient(provider=self.provider, api_key=self.api_key)

    def _call(self, prompt, **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message["content"]

    @property
    def _llm_type(self) -> str:
        return "hf_chat_completion_llm"


def load_llm_model(config):
    llm_params = config["llm_params"]
    return HFChatCompletionLLM(
        model=config["llm_model"],
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        provider=llm_params.get("provider", "nebius"),
        temperature=llm_params.get("temperature", 0.3),
        max_tokens=llm_params.get("max_tokens", 512),
    )
