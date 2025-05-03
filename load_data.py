import json
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def load_data_from_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        json_input = json.load(f)
    return _process_json_input(json_input)


def load_data_from_json_string(json_string):
    json_input = json.loads(json_string)
    return _process_json_input(json_input)


def _process_json_input(json_input):
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
