import pytest
import json
import os
import httpx
from test.test_app import app  # FastAPI app
from app.data.request import RequestData, BookData, UserData


@pytest.mark.asyncio
async def test_recommend_books():
    test_file = os.path.join(os.path.dirname(__file__), "test_data.json")
    with open(test_file, "r", encoding="utf-8") as fr:
        request_dict = json.load(fr)
    
    request_data = RequestData(
        books=[BookData(**book) for book in request_dict["books"]],
        user_data= UserData(**request_dict["user_data"])
    )
    # ASGITransport 사용
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver"
    ) as client:
        response = await client.post("/books/recommendation/", json=request_data.model_dump())

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["recommend_books"], list)
    assert "id" in result["recommend_books"][0]
    assert "AIAnswer" in result["recommend_books"][0]
    
