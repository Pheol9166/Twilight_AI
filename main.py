from fastiapi import FastAPI
from pydantic import BaseModel
from typing import List

from process_data import process_request
from vectorstore import build_vectorstore, build_retriever
from rag_chain import build_rag_chain
from utils import api_login, model_setting, load_config, load_prompt


# 서버 가동 시 로드하여 서비스 중 에러 방지
api_login()
config = load_config()
embedding_model, llm = model_setting(config)
prompt = load_prompt()

app = FastAPI()


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
    books: List[BookData]
    user_data: UserData


class RecommendationResponse(BaseModel):
    id: int  # 책 ID
    AIAnswer: str  # AI 답변


def get_recommendation_func(prompt, embedding, llm, config):
    def generate_recommendation(request):
        user_data, user_answer, documents = process_request(request, config)
        vectorstore = build_vectorstore(documents, embedding)
        retriever = build_retriever(vectorstore, config)
        rag_chain = build_rag_chain(llm, retriever, prompt)
        # ainvoke를 통한 파이프라인 비동기 호출
        result = rag_chain.ainvoke({"query": user_answer, "user_profile": user_data})

        return result

    return generate_recommendation


generate_recommendation = get_recommendation_func(prompt, embedding_model, llm, config)


@app.post("/books/recommendation/")
async def recommend_books(request_data: RequestData):
    recommedations = await generate_recommendation(request_data)
    return recommedations
