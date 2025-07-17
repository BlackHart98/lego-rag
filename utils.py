import typing as t
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass


@dataclass
class SplitResult:
    file_path: str
    file_check_sum: str
    split_check_sum: str
    md_header_split: t.List[float]
    embedding: t.Any | None = None
 
 
@dataclass
class FileMeta:
    file_path: str
    split_count: int
    split_results: t.List[SplitResult]
    
@dataclass    
class MistralModel:
    model_id: str = "mistral"
    embedding_model: str = "mistral-embed"
    llm: str = ""



class AIModel:
    def __init__(self, model_id: str):
        match model_id:
            case MistralModel.model_id:
                 self.model = MistralModel()
            case _:
                raise ValueError("Unsupported model id")
                