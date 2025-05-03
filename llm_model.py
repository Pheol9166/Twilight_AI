from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline


def load_llm_model(config):
    model_name = config['llm_model']
    llm_params = config.get("llm_params", {})
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = llm_params.get("device_map", "auto"),
        dtype = llm_params.get("dtype", "auto")
    )
    
    temperature = llm_params.get("temperature", 0.3)
    top_p = llm_params.get("top_p", 0.9 if temperature >= 0.7 else 0.95)
    top_k = llm_params.get("top_k", 50 if temperature >= 0.7 else 20)
    
    hf_pipeline = pipeline(
        "text-generation",
        model= model,
        tokenizer= tokenizer,
        max_length= llm_params.get("max_tokens", 512),
        temperature= temperature,
        top_p= top_p,
        top_k= top_k,
        repetition_penalty= llm_params.get("repetition_penalty", 1.1)
    )
    
    langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    return langchain_llm