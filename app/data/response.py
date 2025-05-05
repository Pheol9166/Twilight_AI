from pydantic import BaseModel
from typing import List

class ResponseBook(BaseModel):
    id: int
    AIAnswer: str

class RecommendationResponse(BaseModel):
    recommend_books: List[ResponseBook]