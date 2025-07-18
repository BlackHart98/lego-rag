import os
import sys
import asyncio
import typing as t
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from langchain.text_splitter import MarkdownHeaderTextSplitter
from dataclasses import dataclass
from utils import SplitResult, FileMeta, AIModel, FileMetaV2
from config import Config, vector_store
import random

from langchain_mistralai import MistralAIEmbeddings

import hashlib
 
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

import zipfile


class RAGModel:
    _documents: t.List[t.List[Document]] 
    _documents_splits: t.List[Document]
    def __init__(
        self, 
        model_id:str="sample_collection",
        text_spliter:TextLoader=RecursiveCharacterTextSplitter(
            chunk_size=200, 
            chunk_overlap=0),
        vector_store=chromadb.Client(),
    ):
        self._text_split = text_spliter
        self._collection = vector_store.get_or_create_collection(name=model_id)
    
    
    def _local_read_dir(
        self,
        files: t.Set[str],
    ) -> t.List[Document | None]:
        return [TextLoader(item).load() for item in files]  
    
    
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
                file_list = [file_rel_path + "/" + item for item in files]
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
 
 
def filter_threshold(c: chromadb.QueryResult, threshold: float=0.7) -> chromadb.QueryResult:
    return c
    
async def main(argv: t.List[str]) -> int:
    query_result: RAGModel = RAGModel().\
        local_read_dir("local_repo").\
            split_documents().\
                store_embedding().\
                    query_collection(query_texts=["what is my core algorithm?",], n_results=1,)

    print(query_result)
    return 0


if __name__ == "__main__":
    asyncio.run(main(sys.argv))