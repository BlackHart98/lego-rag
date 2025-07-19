import os
import sys
import random
import typing as t
from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader

import pathlib as p
class FileType:
    NIL = ""
    MARKDOWN = ".md"
    TEXT = ".txt"
    PDF = ".pdf"

def _load_file(file_: str, **kwargs) -> t.List[Document]:
    file_extension = p.Path(file_).suffix
    match file_extension:
        case FileType.MARKDOWN | FileType.TEXT | FileType.NIL:
            return TextLoader(file_).load()
        case _:
            raise NotImplementedError(f"File type `{file_}` not supported.")


def get_asym_sleep_time(attempt, base_delay, lag_max) -> float:
    lag_rnd = random.uniform(0, lag_max)
    return (base_delay * (2 ** (attempt - 1))) + lag_rnd