from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
import json


def get_filter_id_func(book_data):
    def filter_id(parsed_output):
        items = parsed_output.get("recommend_books", [])
        recommend = []
        for item in items:
            item_name = item.get("name").strip()
            id = list(filter(lambda x: x["name"] == item_name, book_data))[0]["id"]
            recommend.append({"id": id, "reason": item.get("reason", "")})

        return json.dumps({"recommend_books": recommend})

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
