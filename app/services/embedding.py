from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model(config):
    return HuggingFaceEmbeddings(model_name=config["embedding_model"])
