from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_request(json_input):
    json_input = json_input.dict()
    user_data = json_input["user_data"]
    user_answer = json_input["user_data"]["user_answer"]
    books = json_input["books"]

    return user_data, user_answer, books

def build_documents(books, config):
    documents = [
        Document(
            page_content=book["book_desc"],
            metadata={
                "name": book["name"],
                "author": book["author"],
                "genre": ", ".join(book["gerne"]), # 장르가 List[str]로 구성되어있다 상정
                "publisher": book["publisher"],
            },
        )
        for book in books
    ]
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["splitter"].get("CHUNK_SIZE", 1000),
    chunk_overlap=config["splitter"].get("CHUNK_OVERLAP", 200),
    )
    
    documents = splitter.split_documents(documents)
    
    return documents
    