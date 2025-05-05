from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_request(json_input, config):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["splitter"].get("CHUNK_SIZE", 1000),
        chunk_overlap=config["splitter"].get("CHUNK_OVERLAP", 200),
    )
    user_data = json_input["user"]
    user_answer = json_input["user"]["answer"]
    books = json_input["books"]

    documents = [
        Document(
            page_content=book.book_desc,
            metadata={
                "name": book.name,
                "author": book.author,
                "genre": ", ".join(book.genre), # 장르가 List[str]로 구성되어있다 상정
                "publisher": book.publisher,
            },
        )
        for book in books
    ]

    documents = splitter.split_documents(documents)

    return user_data, user_answer, documents
