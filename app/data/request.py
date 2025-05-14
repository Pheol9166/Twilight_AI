from pydantic import BaseModel
from typing import List
class BookData(BaseModel):
    bookId: int
    name: str
    author: str
    pageCount: int
    description: str

class QuestionAnswer(BaseModel):
    question: str
    userAnswer: str
    matchingTag: str

class UserData(BaseModel):
    memberId: int
    age: int
    gender: str
    personalities: List[str]
    interests: List[str]
    questionAnswers: List[QuestionAnswer]  

class RequestData(BaseModel):
    bookInfo: List[BookData]
    memberInfo: UserData