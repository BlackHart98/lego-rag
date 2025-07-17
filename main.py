import os
import sys
import asyncio
import typing as t
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from langchain.text_splitter import MarkdownHeaderTextSplitter
from dataclasses import dataclass
from utils import SplitResult, FileMeta, AIModel
from config import Config
import random

from langchain_mistralai import MistralAIEmbeddings

import hashlib
    
class RAGModel:
    def __init__(self, model_id: str, api_key: str=Config.MISTRAL_API_KEY) -> None:
        print(Config.MISTRAL_API_KEY)
        ai_model = AIModel(model_id).model
        self._embeddings = MistralAIEmbeddings(
            model=ai_model.embedding_model,
            api_key=api_key,
        )
        
    async def get_mdfile_embeddings(
        self,
        files: t.Set[str], 
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")]
    ) -> t.List[FileMeta] | None:
        result: t.List[FileMeta] = []
        for item in files:
            with open(item, "r") as f:
                markdown_document: str = f.read()
                f.close()
            file_check_sum: str = hashlib.md5(markdown_document.encode()).hexdigest()
            markdown_splitter: MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter(headers_to_split_on)
            md_header_splits: SplitResult = [
                SplitResult(
                    file_path=item, 
                    file_check_sum=file_check_sum,
                    split_check_sum=hashlib.md5(s.page_content.encode()).hexdigest(),
                    md_header_split=s) 
                for s in markdown_splitter.split_text(markdown_document)]
            corountines = [self._generate_embedding(idx, item) for idx, item in enumerate(md_header_splits)]
            split_results: t.List[SplitResult] = await asyncio.gather(*corountines)
            result += [FileMeta(
                file_path=item,
                split_count=len(md_header_splits),
                split_results=split_results
            )]
        return result


    async def _generate_embedding(self, index: int, chunk: SplitResult) -> SplitResult:
        delay = random.uniform(0, 0.5)
        # print(f"generating embedding for {chunk.split_check_sum}")
        # print(f"chunk: #{index} in file {chunk.file_path} is sleeping for {delay}")
        # await asyncio.sleep(delay)
        # print(f"chunk #{index} is done.")
        return SplitResult(
            file_path=chunk.file_path, 
            file_check_sum=chunk.file_check_sum,
            split_check_sum=chunk.split_check_sum,
            embedding=self._embeddings.embed_query(chunk.md_header_split.page_content),
            md_header_split=chunk.md_header_split
        )

 
    async def local_read_dir(self, directory: str) -> t.List[FileMeta]:
        local_repo = os.walk(directory)
        file_list: str = []
        for folder_triple in local_repo:
            file_rel_path, _, files = folder_triple
            file_list += [file_rel_path + "/" + item for item in files]
        result: t.List[FileMeta] | None = await self.get_mdfile_embeddings(file_list)
        return result


async def main(argv: t.List[str]) -> int:
    foobar: t.List[FileMeta] = await RAGModel("mistral").local_read_dir("local_repo")
    print(foobar)
    return 0


if __name__ == "__main__":
    asyncio.run(main(sys.argv))