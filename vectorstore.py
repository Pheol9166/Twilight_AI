from langchain.vectorstores import FAISS
from typing import Dict, Any, Optional


def build_vectorstore(documents, embedding_model):
    return FAISS.from_documents(documents, embedding_model)


def build_retriever(
    vectorstore,
    type: str = "similarity",
    search_kwargs: Optional[Dict[str, Any]] = None,
):
    return vectorstore.as_retriever(search_type=type, search_kwargs=search_kwargs)
