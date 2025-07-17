import asyncio
import typing as t
from pydantic import BaseModel


class RAGModel(BaseModel):
    _ai_model: t.Any
    # _content: str
    # def __init__(self, ai_model: t.Any, ):
    #     pass
    
    # async def embedded(self, content: str) -> t.Any:
    #     return ""
    
    # async def response(self, content: str) -> t.Any:
    #     return ""
    
    # async def 