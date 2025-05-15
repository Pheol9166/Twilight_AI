import redis.asyncio as redis
import redis.exceptions
import json
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.data.request import RequestData
from app.data.response import RecommendationResponse
from app.services.process_data import split_request, build_documents
from app.services.vectorstore import build_vectorstore, build_retriever
from app.services.rag_chain import build_rag_chain
from app.startup import initialize_app
import httpx
import os


redis_client = redis.Redis(host="localhost", port=6379, db=0)

AI_AUTH_TOKEN = os.environ.get("AI_AUTH_TOKEN")
BACKEND_URL = os.environ.get("BACKEND_URL")
STREAM_KEY = "ai:recommend"
GROUP_NAME = "ai-consumers"


async def send_result_to_backend(result: RecommendationResponse):
    headers = {"X-AI-AUTH-TOKEN": AI_AUTH_TOKEN, "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            BACKEND_URL, json=result.model_dump(), headers=headers
        )

    if response.status_code == 200:
        print("전송 성공")
    else:
        print(f"전송 실패: {response.status_code}, {response.text}")


async def consume_redis_stream(redis_client: redis.Redis):
    while True:
        try:
            redis_client.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=True)
        except redis.exceptions.ResponseError:
            pass
        except redis.exceptions.ConnectionError:
            print("Redis 서버 연결 실패: 서버가 실행 중인지 확인하세요.")
            return "Redis 서버 연결 실패"        

        try:
            messages = redis_client.xreadgroup(
                groupname=GROUP_NAME,
                consumername="consumer1",
                streams={STREAM_KEY: ">"},
                count=1,
                block=0,
            )

            for message in messages:
                _, data = message
                for msg_id, payload in data:
                    payload_data = json.loads(payload[b"payload"].decode("utf-8"))
                    print("받은 데이터:", payload_data)

                    await redis_client.xack(STREAM_KEY, GROUP_NAME, msg_id)

                    request_data = RequestData(**payload_data)
                    result: RecommendationResponse = await generate_recommendation(
                        request_data
                    )
                    result.member_id = request_data.memberInfo.memberId

                    await send_result_to_backend(result)
        except redis.exceptions.ConnectionError as e:
            print(f"Redis connection error: {e}, 5초 후 재시도")
            await asyncio.sleep(5)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(consume_redis_stream(redis_client))
    yield


config, prompt, embedding_model, llm = initialize_app(
    config_path="app/config/config.json", prompt_path="app/config/prompt.txt"
)


app = FastAPI(lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
