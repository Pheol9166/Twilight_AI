from pydantic import BaseModel

class RecommendationResponse(BaseModel):
    member_id: int
    book_id: int
    AI_answer: str