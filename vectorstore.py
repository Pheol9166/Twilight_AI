from langchain.vectorstores import FAISS


def build_vectorstore(documents, embedding_model):
    return FAISS.from_documents(documents, embedding_model)


def build_retriever(vectorstore, config):
    return vectorstore.as_retriever(
        search_type=config["retriever"].get("search_type", "similarity"),
        search_kwargs=config["retriever"].get("search_kwargs", {"k": 5})
    )
