import os
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import chromadb


DOT_ENV_PATH = find_dotenv()
load_dotenv(dotenv_path=DOT_ENV_PATH, override=True)


class Config:
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")
    
    
vector_store = chromadb.Client()