import os
import sys
import asyncio
import typing as t

from core import (
    RAGModel, 
    Questionnaire, 
    Aggregator, 
    AggregatedQueryResult,
    ResponseGenerator)        
from langchain_mistralai import ChatMistralAI

 
async def main(argv: t.List[str]) -> int:
    rag_model = RAGModel().\
        local_read_dir("local_repo").\
            split_documents().\
                store_embedding(namespace="my_collection")
                
    query: t.List[str] = Questionnaire(query="what is my core algorithm?",).\
        generate_retrival_query().\
            get_query_splits()
    
    result: t.List[AggregatedQueryResult] = Aggregator(
        rag_model.query_collection(
            query_texts=query, 
            n_results=2,)
        ).merge_query_results()
    
    response = await ResponseGenerator(
        query,
        llm=ChatMistralAI(
            model="mistral-large-latest", 
            mistral_api_key=os.getenv("MISTRAL_API_KEY"), 
            temperature=0,
        )
    ).generate_response(result)
    
    print(response.query_response)
    print(response.citations)
    
    return 0


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
