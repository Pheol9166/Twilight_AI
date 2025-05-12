from fastapi import APIRouter
from app.data.request import RequestData
from app.data.response import RecommendationResponse
from app.services.process_data import split_request, build_documents
from app.services.vectorstore import build_vectorstore, build_retriever
from app.services.rag_chain import build_rag_chain
import os
import httpx


AI_AUTH_TOKEN = os.environ.get("AI_AUTH_TOKEN")
BACKEND_URL = os.environ.get("BACKEND_URL")

async def send_result_to_backend(result: RecommendationResponse):
    headers = {
        "X-AI-AUTH-TOKEN": AI_AUTH_TOKEN,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(BACKEND_URL, json=result, headers=headers)
    if response.status_code == 200:
        print("전송 성공")
    else:
        print(f"전송 실패: {response.status_code}, {response.text}")
    
def build_routes(app, config, prompt, embedding_model, llm):
    router = APIRouter()

    async def generate_recommendation(request_data: RequestData):
        user_data, tags, qna, books = split_request(request_data)
        documents = build_documents(books, config)
        vectorstore = build_vectorstore(documents, embedding_model)
        retriever = build_retriever(vectorstore, config)
        rag_chain = build_rag_chain(llm, retriever, prompt, books)
        result = await rag_chain.ainvoke(
            {"query": tags, "user_profile": user_data, "qna": qna}
        )
        return result
    
    @router.post("/books/recommendation/", response_model=RecommendationResponse)
    async def recommend_books(request_data: RequestData):
        result: RecommendationResponse = await generate_recommendation(request_data)
        result.member_id = request_data.user_data.id
        return send_result_to_backend(result.model_dump())

    app.include_router(router)
