from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
import json


def build_rag_chain(llm, retriever, prompt_text, book_db):
    parser = JsonOutputParser()
    
    prompt = PromptTemplate(
        input_variables=["user_profile", "question", "context"], 
        template=prompt_text
    )
    
    def get_filter_books_func(book_db):
        def filter_books(parsed_output):
            # parsed_output -> {"recommend_books": [{"name": "책A", "reason": "추천 이유 1~2문장"}, {"name": "책B", "reason": "추천 이유 1~2문장"}]}
            print(f"parsed_output: \n{parsed_output}")
            items = parsed_output.get("recommend_books", [])
            recommend = []
            
            for item in items:
                item_name = item.get("name").strip()
                item_reason = item.get("reason", "").strip()
                matched_book = book_db[book_db['name'] == item_name].copy()
                if not matched_book.empty:
                    matched_book["reason"] = item_reason
                    # JSON 입력 시 datetime인지 문자열인지 확인 필요
                    matched_book['pub_date'] = matched_book['pub_date'].dt.strftime('%Y-%m-%d')
                    matched_book = matched_book.to_dict()                  
                    recommend.append(matched_book)
                
            return json.dumps({"recommended_books": recommend}, ensure_ascii=False, indent=2)
        return filter_books
                
    filter_books = get_filter_books_func(book_db)
    
    rag_chain = (
        {
            "context": itemgetter("query") | retriever,
            "question": itemgetter("query"),
            "user_profile": itemgetter("user_profile"),
        }
        | prompt
        | llm
        | parser
        | filter_books
    )
    return rag_chain