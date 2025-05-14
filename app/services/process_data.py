from typing import List
from app.data.request import BookData, UserData, QuestionAnswer, RequestData
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_request(request: RequestData):
    user_data: UserData = request.memberInfo
    user_answer: List[QuestionAnswer] = request.memberInfo.questionAnswers
    user_data.pop("questionAnswers", None)  # user 데이터에서 질문 & 답변 제거
    tags: str = " ".join([answer.matchingTag for answer in user_answer])
    qna = [{answer.question: answer.userAnswer} for answer in user_answer]
    books: List[BookData] = request.bookInfo

    return user_data, tags, qna, books

def build_documents(books: List[BookData], config):
    documents = [
        Document(
            page_content=book["description"],
            metadata={
                "id": book["bookId"],
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
    