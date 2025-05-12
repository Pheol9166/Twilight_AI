from pydantic import BaseModel
from typing import List, Optional

class BookData(BaseModel):
    id: int
    name: str
    author: str
    page_count: Optional[int] = None
    book_desc: str

class QuestionAnswer(BaseModel):
    question: str
    user_answer: str
    match_tag: str

class UserData(BaseModel):
    id: int
    age: int
    gender: str
    personality: str
    interests: List[str]
    read_books: List[str]  
    user_answer: List[QuestionAnswer]  

class RequestData(BaseModel):
    books: List[BookData]
    user_data: UserData

