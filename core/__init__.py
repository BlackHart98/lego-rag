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


class QueryStrategy(IntEnum):
    NO_STRATEGY: int = 1
    REPHRASE_STRATEGY: int = 2
    SEGMENTATION_STRATEGY: int = 3
    


class RAGModel:
    _documents: t.List[t.List[Document]] 
    _documents_splits: t.List[Document]
    def __init__(
        self, 
        model_id:str="sample_collection",
        text_spliter:TextSplitter=RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=0),
        vector_store=vector_store,
    ):
        self._text_split = text_spliter
        self._collection = vector_store.get_or_create_collection(
            name=model_id,
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
        
    def query_collection(self, **kwargs) -> chromadb.QueryResult:
        return self._collection.query(**kwargs)
    
    
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
    _query_results = []  
    _top_k: int | None = None
    
    def __init__(self, query_results: t.List[chromadb.QueryResult], top_k:int | None=None):
        self._query_results = query_results
        self._top_k = top_k
        
    def merge_query_results(self) -> chromadb.QueryResult:
        if self._top_k:
            for q in self._query_results:
                print(q)
        else:
            pass
    
    def _merge_query_result(self, query_result: chromadb.QueryResult) -> chromadb.QueryResult:
        # create a faux mapping between  the {array_index}_id_{idx} and the distance
        pass