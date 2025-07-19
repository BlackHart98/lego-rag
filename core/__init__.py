import os
import sys
import typing as t
from pydantic import BaseModel
from config import vector_store

import zipfile 
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter, TextSplitter
from enum import IntEnum

import pathlib as p
from utils import _load_file
import logging


Metadata: t.TypeAlias = t.Mapping[str, t.Optional[t.Union[str, int, float, bool]]]

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



class RAGModel:
    _documents: t.List[t.List[Document]] 
    _documents_splits: t.List[Document]
    def __init__(
        self, 
        namespace:str="sample_collection",
        text_spliter:TextSplitter=RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=0),
        vector_store=vector_store,
    ):
        self._text_split = text_spliter
        self._collection = vector_store.get_or_create_collection(
            name=namespace,
            metadata={"hnsw:space": "cosine"},
            )
    
    
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
    
    def local_read_dir(self, directory: str):
        try:
            local_repo = os.walk(directory)
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
    
    def store_embedding(self, id_prefix="id"):
        self._collection.upsert(
            documents=[item.page_content for item in self._documents_splits],
            ids=[f"{id_prefix}_{idx}" for idx, _ in enumerate(self._documents_splits)],
            metadatas=[item.metadata for item in self._documents_splits]
        )
        return self
        
    def query_collection(self, routing_to_namespace: bool=False, **kwargs) -> chromadb.QueryResult:
        if "query_texts" in kwargs and routing_to_namespace:
            kwargs["query_texts"] = kwargs["query_texts"]
        return self._collection.query(**kwargs)
    
    
    def route_queery_to_namespace(self, query_text: str) -> str:
        raise NotImplementedError()  



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
    
    def __init__(self, query_results: chromadb.QueryResult, threshold: float | None=None):
        self._query_results = query_results
        self._threshold = threshold
        
    def merge_query_results(self) -> t.List[AggregatedQueryResult]:
        result: t.List[AggregatedQueryResult] = []
        n_for_queries = len(self._query_results["ids"])
        if not (self._query_results["documents"]) or n_for_queries <= 0 or not(self._query_results["metadatas"]):
            return []
        else:
            for idx in range(n_for_queries):
                n_for_results = len(self._query_results["ids"][idx])
                if n_for_results <= 0:
                    break
                else:
                    for jdx in range(n_for_results):
                        result += [AggregatedQueryResult(
                            id=self._query_results["ids"][idx][jdx], 
                            cos_distance=self._query_results["distances"][idx][jdx] if self._query_results["distances"] else 0,
                            cos_similarity=(1 - self._query_results["distances"][idx][jdx]) if self._query_results["distances"] else 1,
                            document=self._query_results["documents"][idx][jdx],
                            metadata=self._query_results["metadatas"][idx][jdx],)]
            if self._threshold:
                return [item for item in result if item.cos_similarity >= self._threshold]
            else:
                return result
    
    def _deduplicate_result(self, result: t.List[AggregatedQueryResult]) -> t.List[AggregatedQueryResult]:
        raise NotImplementedError()  