from fastapi import APIRouter
from request import RequestData, RecommendationResponse
from process_data import split_request, build_documents
from vectorstore import build_vectorstore, build_retriever
from rag_chain import build_rag_chain


def build_routes(app, config, prompt, embedding_model, llm):
    router = APIRouter()

    async def generate_recommendation(request_data: RequestData):
        user_data, user_answer, books = split_request(request_data)
        documents = build_documents(books, config)
        vectorstore = build_vectorstore(documents, embedding_model)
        retriever = build_retriever(vectorstore, config)
        rag_chain = build_rag_chain(llm, retriever, prompt, books)
        result = await rag_chain.ainvoke(
            {"query": user_answer, "user_profile": user_data}
        )
        return result

    @router.post("/books/recommendation/", response_model=RecommendationResponse)
    async def recommend_books(request_data: RequestData):
        return await generate_recommendation(request_data)

    app.include_router(router)
