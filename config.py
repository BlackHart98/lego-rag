import os
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import chromadb


DOT_ENV_PATH = find_dotenv()
load_dotenv(dotenv_path=DOT_ENV_PATH, override=True)


class Config:
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")
    DEFAULT_RETRY_COUNT = 1
    DEFAULT_DELAY_SECONDS = 2.0
    DEFAULT_LAG_MAX_SECONDS = 0.5
    
# I need to change this to make this such that I can easily switch between the different chromadb mode    
vector_store = chromadb.Client()
