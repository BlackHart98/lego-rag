import sys
import asyncio
import typing as t

from core import RAGModel, Questionnaire, QueryStrategy        


# I should use pathlib to handle paths 
async def main(argv: t.List[str]) -> int:
    query: t.List[str] = Questionnaire(
        query="what is my core algorithm?", 
        query_strategy=QueryStrategy.REPHRASE_STRATEGY, 
        query_split_count=1,).\
        generate_retrival_query().\
            get_query_splits()
            
    query_result = RAGModel().\
        local_read_dir("local_repo").\
            split_documents().\
                store_embedding().\
                    query_collection(query_texts=query, n_results=2,)
    print(query_result)
    return 0


if __name__ == "__main__":
    asyncio.run(main(sys.argv))