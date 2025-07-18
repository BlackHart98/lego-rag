import typing as t
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
from enum import IntEnum


class FileType:
    NIL = ""
    MARKDOWN = ".md"
    TEXT = ".txt"
    PDF = ".pdf"


class QueryStrategy(IntEnum):
    BASIC_STRATEGY: int = 1
    STEP_BACK_STRATEGY: int = 2
    REPHRASE_STRATEGY: int = 3
    