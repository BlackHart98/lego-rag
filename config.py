from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

DOT_ENV_PATH = find_dotenv()
load_dotenv(dotenv_path=DOT_ENV_PATH, override=True)


class Config(BaseModel):
    pass