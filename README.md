# lego-rag
*A very simple RAG scaffolding*

## Prerequisites
1. Install Python3 (python 3.11+ preferably)

## How to use this
1. Create a `.env` file to keep your credentials, which are
```env
MISTRAL_AI_KEY=
```
> [!Note]
> I am currently using Mistral API


## Example (tentative)

```python

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
    my_sample_collection = RAGModel().\
        local_read_dir("local_repo").\
            split_documents().\
                store_embedding(namespace="my_collection")
                
    query: t.List[str] = Questionnaire(query="what is my core algorithm?",).\
        generate_retrival_query().\
            get_query_splits()
            
    query_result = my_sample_collection.query_collection(query_texts=query, n_results=2,)
    
    result : t.List[AggregatedQueryResult] = Aggregator(query_result).merge_query_results()
    
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
```
> [!Note]
> - For now the primary vector database is Chroma
> - Still an active work in progress