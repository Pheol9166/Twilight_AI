from pydantic import BaseModel
from typing import List, Dict, Any

class BookData(BaseModel):
    id: int
    name: str
    author: str
    publisher: str
    gerne: List[str]
    book_desc: str


class UserData(BaseModel):
    # id: int
    age: int
    gender: str
    personality: str
    interests: List[str]
    read_books: List[int]  # 읽은 책들의 ID 목록
    user_answer: str  # 질문에 대한 답변


class RequestData(BaseModel):
    books: List[Dict[Any, Any]]
    user_data: Dict[Any, Any]

class ResponseBook(BaseModel):
    id: int
    AIAnswer: str

class RecommendationResponse(BaseModel):
    recommend_books: List[ResponseBook]