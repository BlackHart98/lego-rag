import typing as t
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass


@dataclass
class SplitResult:
    file_path: str
    file_check_sum: str
    split_check_sum: str
    md_header_split: t.Any
    embedding: t.Any | None = None
 
 
@dataclass
class FileMeta:
    file_path: str
    split_count: int
    split_results: t.List[SplitResult]