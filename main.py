import sys
import asyncio
import typing as t

from core import RAGModel, Questionaire
        
    
async def main(argv: t.List[str]) -> int:
    query_result: RAGModel = RAGModel().\
        local_read_dir("local_repo").\
            split_documents().\
                store_embedding().\
                    query_collection(query_texts=["what is my core algorithm?",], n_results=1,)
    print(query_result)
    
    sample_questionaire = Questionaire(query="what is my core algorithm?").\
        generate_retrival_query().\
            get_query_splits()
    print(sample_questionaire)
    return 0


if __name__ == "__main__":
    asyncio.run(main(sys.argv))