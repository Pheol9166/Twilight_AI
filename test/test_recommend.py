from app.startup import initialize_app
from app.services.process_data import split_request, build_documents
from app.services.vectorstore import build_vectorstore, build_retriever
from app.services.rag_chain import build_rag_chain
from app.data.request import RequestData, BookData, UserData
import json

with open("./test/test_data.json", "r", encoding="utf-8") as fr:
    request_dict = json.load(fr)

request_data = RequestData(
    bookInfo=[BookData(**book) for book in request_dict["bookInfo"]],
    memberInfo= UserData(**request_dict["memberInfo"])
)


config, prompt, embedding_model, llm = initialize_app(
    config_path="app/config/config.json", prompt_path="app/config/prompt.txt"
)

user_data, tags, qna, books = split_request(request_data)
documents = build_documents(books, config)
vectorstore = build_vectorstore(documents, embedding_model)
retriever = build_retriever(vectorstore, config)
rag_chain = build_rag_chain(llm, retriever, prompt, books)
result = rag_chain.invoke(
    {"query": tags, "user_profile": user_data, "qna": qna}
)
print(result)
    