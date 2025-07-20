import os
import sys
import typing as t
from pydantic import BaseModel
from config import vector_store, Config

import zipfile 
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from enum import IntEnum
from jinja2 import Environment, FileSystemLoader, Template

import pathlib as p
from utils import _load_file, get_asym_sleep_time
import logging
import asyncio

from chromadb import Collection


Metadata: t.TypeAlias = t.Mapping[str, t.Optional[t.Union[str, int, float, bool]]]
T = t.TypeVar('T')

class QueryStrategy(IntEnum):
    NO_STRATEGY: int = 1
    REPHRASE_STRATEGY: int = 2
    SEGMENTATION_STRATEGY: int = 3
    


class AggregatedQueryResult(BaseModel):
    id : str
    cos_distance: float # range 0 to 2
    cos_similarity: float # range -1 to 1
    document: str
    metadata: Metadata
    

class QueryResponse(BaseModel):
    model_id : str | None
    query_response: str
    citations: t.List[AggregatedQueryResult] | None



class RAGModel:
    _documents: t.List[t.List[Document]] 
    _documents_splits: t.List[Document]
    def __init__(
        self, 
        text_spliter:TextSplitter=RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=0),
        vector_store=vector_store,
    ):
        self._text_split = text_spliter
        self._collections: Collection = {}
    
    
    def _local_read_dir(
        self,
        files: t.Set[str],
        fn = lambda x: True
    ) -> t.List[Document | None]:
        return [_load_file(item) for item in files if fn(item)]  
    
    def local_read_zip(self, zip_file_path: str):
        with zipfile.ZipFile(zip_file_path) as zf:
            extracted_files = zf
        print(zip_file_path)
        return self
   
    def remote_read_dir(self,):
        return self

    def read_webpage(self,):
        return self

    def local_read_dir(self, directory: str):
        try:
            local_repo = os.walk(p.Path(directory).absolute())
            file_list: str = {}
            result = []
            for folder_triple in local_repo:
                file_rel_path, _, files = folder_triple
                file_list = [file_rel_path + "/" + item for item in files] # I need to address the as soon as possible dumbass
                result += self._local_read_dir(file_list)
            self._documents = result
        except Exception as e:
            raise ValueError(f"Fail to read direction due ot {e}")
        return self
    
    def split_documents(self):
        result = []
        for item in self._documents:
            result += self._text_split.split_documents(item)
        self._documents_splits = result
        return self
    
    def store_embedding(self, namespace: str="sample_collection", id_prefix="id"):
        collection = vector_store.get_or_create_collection(
            name=namespace,
            metadata={"hnsw:space": "cosine"},)
        self._collection = collection.upsert(
            documents=[item.page_content for item in self._documents_splits],
            ids=[f"{id_prefix}_{idx}" for idx, _ in enumerate(self._documents_splits)],
            metadatas=[item.metadata for item in self._documents_splits]
        )
        self._collections[namespace] = collection
        return self
    
    def cluster_embeddings(
        self,
        namespaces:t.List[str] | None=[], 
        clustering_fn: t.Callable[..., t.Dict[str, T]] | None=None
    ):
        raise ValueError("The function is not yet implemented") 
        
    def query_collection(self, routing_to_namespace: bool=False, **kwargs) -> t.List[chromadb.QueryResult]:
        if "query_texts" in kwargs and routing_to_namespace:
            kwargs["query_texts"] = kwargs["query_texts"]
        return [self._collections[namespace].query(**kwargs) for namespace in self._collections]
    
    def route_query_to_namespace(self, query_text: str) -> str:
        raise ValueError("The function is not yet implemented")   



class Questionnaire(BaseModel):
    query: str
    query_strategy: QueryStrategy = QueryStrategy.NO_STRATEGY
    query_split_count: int = 0
    query_splits: t.List[str] = []
    
    # I need to look for a better abstraction
    def generate_retrival_query(
        self, 
        rephrase_template:str | None="../prompt_templates/rephrase.txt",
    ):
        match self.query_strategy:
            case QueryStrategy.NO_STRATEGY:
                if self.query_split_count > 0:
                    logging.warning("`query_split_count` for `NO_STRATEGY` can only be 0")
                self.query_splits = [self.query]
                return self
            case QueryStrategy.REPHRASE_STRATEGY:
                self.query_splits = [self.query]
                return self
            case QueryStrategy.SEGMENTATION_STRATEGY:
                splits: t.List[str] = self.query.split()
                count = 0
                self.query_splits = [self.query]
                for i in range(1, len(splits)):
                    if (count > self.query_split_count) or (len(splits[i:]) <= 1):
                        break
                    else:
                        self.query_splits += [" ".join(splits[i:])]
                        count += 1
                return self
            case _:
                self.query_splits = [self.query]
                return self
    
    def get_query_splits(self) -> t.List[str]:
        return self.query_splits



class Aggregator:  
    _threshold: float | None = None
    
    def __init__(self, query_results: t.List[chromadb.QueryResult], threshold: float | None=.5):
        self._query_results = query_results
        self._threshold = threshold
        
    def merge_query_results(self) -> t.List[AggregatedQueryResult]:
        nested_query_result: t.List[t.List[AggregatedQueryResult]] = [self._merge_query_result(item) for item in self._query_results]
        flattened_query_results = [x for item in nested_query_result for x in item]
        return flattened_query_results
    
    def _merge_query_result(self, query_result) -> t.List[AggregatedQueryResult]:
        result: t.List[AggregatedQueryResult] = []
        n_for_queries = len(query_result["ids"])
        if not (query_result["documents"]) or n_for_queries <= 0 or not(query_result["metadatas"]):
            return []
        else:
            for idx in range(n_for_queries):
                n_for_results = len(query_result["ids"][idx])
                if n_for_results <= 0:
                    break
                else:
                    for jdx in range(n_for_results):
                        result += [AggregatedQueryResult(
                            id=query_result["ids"][idx][jdx], 
                            cos_distance=query_result["distances"][idx][jdx] if query_result["distances"] else 0,
                            cos_similarity=(1 - query_result["distances"][idx][jdx]) if query_result["distances"] else 1,
                            document=query_result["documents"][idx][jdx],
                            metadata=query_result["metadatas"][idx][jdx],)]
            if self._threshold:
                return [item for item in result if item.cos_similarity >= self._threshold]
            else:
                return result
    
    def _deduplicate_result(self, result: t.List[AggregatedQueryResult]) -> t.List[AggregatedQueryResult]:
        raise NotImplementedError()  
    
    
class ResponseGenerator:
    _llm: BaseChatModel | None = None
    def __init__(
        self, 
        query:t.List[str],
        llm: BaseChatModel | None = None, 
        role_template_file:str="prompt_templates/rag_role.txt",
        response_template_file:str="prompt_templates/response.txt",
    ):
        self._llm = llm
        self._query = query
        try:
            with open(p.Path(response_template_file).absolute(), "r") as f, open(p.Path(role_template_file).absolute(), "r") as rf:
                content = f.read()
                role_content = rf.read()
                f.close(); rf.close()
            role_template = Template(role_content)
            self._system_role = ("system", role_template.render({}))
            self._response_template = Template(content)
        except Exception as e:
            raise ValueError(f"Failed to initialize `ResponseGenerator` object due to {e}")
        
    async def generate_response(
        self, 
        aggregate_result: t.List[AggregatedQueryResult],
        model_id: str=None, 
        fallback_response: str="I don't know.",
        retry:int=Config.DEFAULT_RETRY_COUNT,
        base_delay:float=Config.DEFAULT_DELAY_SECONDS,
        lag_max:float=Config.DEFAULT_LAG_MAX_SECONDS,
    ) -> QueryResponse:
        if self._llm:
            if len(aggregate_result) <= 0:
                return QueryResponse(
                    model_id=model_id,
                    query_response=fallback_response,
                    citations=None
                )
            else:
                content_list = [{"page_content" : item.document} for item in aggregate_result]
                citations = {"citations": content_list, "queries" : [{"query" : item} for item in self._query]}
                body = self._response_template.render(citations)
                print(body)
                prompt_template = ChatPromptTemplate.from_messages([
                    self._system_role,
                    ("user", "{body}")
                ])
                prompt = prompt_template.format(body=body)
                for i in range(retry + 1):
                    try:
                        response = await self._llm.ainvoke(prompt)
                        return QueryResponse(
                            model_id=model_id,
                            query_response=response.content,
                            citations=aggregate_result,
                        )
                    except Exception as e:
                        logging.error(f"Failed to summarize answer due to {e}.")
                        if retry <= i: break
                        sleep_time = get_asym_sleep_time(i + 1, base_delay, lag_max)  
                        logging.error(f"Sleeping for {sleep_time:.2f}s before retrying…")
                        await asyncio.sleep(sleep_time)                        
                return QueryResponse(
                    model_id=model_id,
                    query_response=fallback_response,
                    citations=None,
                )
        else:
            raise ValueError("No chat model specified.")
