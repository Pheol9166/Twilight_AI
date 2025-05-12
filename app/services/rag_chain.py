from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from app.data.response import RecommendationResponse


def get_filter_id_func(book_data):
    name_to_id = {book["name"]: book["id"] for book in book_data}

    def filter_id(parsed_output):
        name = parsed_output['name']
        if name in name_to_id:
            book_id = name_to_id.get(name)
        if book_id is None:
            return RecommendationResponse(book_id=0, AI_answer=parsed_output.get("reason", ""), member_id= 0)
        else:
            return RecommendationResponse(book_id=book_id, AI_answer=parsed_output.get("reason", ""), member_id= 0)

    return filter_id


def build_rag_chain(llm, retriever, prompt_text, book_data):
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        input_variables=["user_profile", "qna", "context"], template=prompt_text
    )

    filter_id = get_filter_id_func(book_data)

    rag_chain = (
        {
            "context": itemgetter("query") | retriever,
            "question": itemgetter("qna"),
            "user_profile": itemgetter("user_profile"),
        }
        | prompt
        | llm
        | parser
        | filter_id
    )
    return rag_chain
