from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from request import ResponseBook


def get_filter_id_func(book_data):
    name_to_id = {book["name"]: book["id"] for book in book_data}

    def filter_id(parsed_output):
        items = parsed_output.get("recommend_books", [])
        recommend = []
        for item in items:
            item_name = item.get("name", "").strip()
            book_id = name_to_id.get(item_name)
            if book_id is None:
                recommend.append(ResponseBook(id="", AIAnswer=item.get("reason", "")))
            else:
                recommend.append(ResponseBook(id=book_id, AIAnswer=item.get("reason", "")))

        return {"recommend_books": recommend}

    return filter_id


def build_rag_chain(llm, retriever, prompt_text, book_data):
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        input_variables=["user_profile", "question", "context"], template=prompt_text
    )

    filter_id = get_filter_id_func(book_data)

    rag_chain = (
        {
            "context": itemgetter("query") | retriever,
            "question": itemgetter("query"),
            "user_profile": itemgetter("user_profile"),
        }
        | prompt
        | llm
        | parser
        | filter_id
    )
    return rag_chain
