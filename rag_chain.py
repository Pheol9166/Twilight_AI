from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter


def build_rag_chain(llm, retriever, prompt_text):
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        input_variables=["user_profile", "question", "context"], template=prompt_text
    )

    rag_chain = (
        {
            "context": itemgetter("query") | retriever,
            "question": itemgetter("query"),
            "user_profile": itemgetter("user_profile"),
        }
        | prompt
        | llm
        | parser
    )
    return rag_chain
