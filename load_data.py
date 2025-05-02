import json
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def load_data_from_json_file(file_path):
    """JSON 파일에서 데이터 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        json_input = json.load(f)
    return _process_json_input(json_input)


def load_data_from_json_string(json_string):
    """JSON 문자열에서 데이터 로드 (Gradle CLI에서 받는 경우)"""
    json_input = json.loads(json_string)
    return _process_json_input(json_input)


def _process_json_input(json_input):
    """공통 처리 로직"""
    user = json_input["user"]
    answer = json_input["answer"]
    books = json_input["books"]
    documents = [
        Document(
            page_content=book.get("book_desc", ""),
            metadata={
                "name": book.get("name", ""),
                "author": book.get("author", ""),
                "genre": book.get("genre", ""),
            },
        )
        for book in books
    ]
    return user, answer, documents


def split_documents(documents):
    """DB 문서 청크 분리"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
