import os
import sys
import asyncio
import typing as t
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from langchain.text_splitter import MarkdownHeaderTextSplitter
from dataclasses import dataclass
from utils import SplitResult, FileMeta
import random

import hashlib
# from core import *
    

async def get_mdfile_embeddings(
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
        corountines = [generate_embedding(idx, item) for idx, item in enumerate(md_header_splits)]
        split_results: t.List[SplitResult] = await asyncio.gather(*corountines)
        result += [FileMeta(
            file_path=item,
            split_count=len(md_header_splits),
            split_results=split_results
        )]
    return result


# simulate embedding generation
async def generate_embedding(index: int, chunk: SplitResult) -> SplitResult:
    delay = random.uniform(0, 0.5)
    print(f"generating embedding for {chunk.split_check_sum}")
    print(f"chunk: #{index} in file {chunk.file_path} is sleeping for {delay}")
    await asyncio.sleep(delay)
    print(f"chunk #{index} is done.")
    return SplitResult(
        file_path=chunk.file_path, 
        file_check_sum=chunk.file_check_sum,
        split_check_sum=chunk.split_check_sum,
        embedding=None,
        md_header_split=chunk.md_header_split
    )


# text only: 
async def local_read_dir(directory: str) -> None:
    local_repo = os.walk(directory)
    file_list: str = []
    for folder_triple in local_repo:
        file_rel_path, _, files = folder_triple
        file_list += [file_rel_path + "/" + item for item in files]
    result: t.List[FileMeta] | None = await get_mdfile_embeddings(file_list)
    print(result)
    

async def main(argv: t.List[str]) -> int:
    await local_read_dir("local_repo")
    return 0


if __name__ == "__main__":
    asyncio.run(main(sys.argv))