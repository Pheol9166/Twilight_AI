from typing import List, Dict
from app.data.request import BookData, UserData, QuestionAnswer
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_request(json_input):
    json_input: Dict = json_input.dict()
    user_data: UserData = json_input["user_data"]
    user_answer: List[QuestionAnswer] = json_input["user_data"]["user_answer"]
    user_data.pop("user_answer", None)  # user 데이터에서 질문 & 답변 제거
    tags: str = " ".join([answer['match_tag'] for answer in user_answer])
    qna = [{answer['question']: answer['user_answer']} for answer in user_answer]
    books: List[BookData] = json_input["books"]

    return user_data, tags, qna, books

def build_documents(books: List[BookData], config):
    documents = [
        Document(
            page_content=book["book_desc"],
            metadata={
                "id": book["id"],
                "name": book["name"],
                "author": book["author"]
            },
        )
        for book in books
    ]
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["splitter"].get("CHUNK_SIZE", 1000),
    chunk_overlap=config["splitter"].get("CHUNK_OVERLAP", 200),
    )
    
    documents = splitter.split_documents(documents)
    
    return documents
    