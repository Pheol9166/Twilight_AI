from langchain.embeddings import HuggingFaceEmbeddings


def load_embedding_model(config):
    return HuggingFaceEmbeddings(model_name=config["embedding_model"])
