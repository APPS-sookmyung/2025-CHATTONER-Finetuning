import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

BASE_MODEL_ID = "google/gemma-2-2b-it"
LORA_MODEL_ID = "jjejieun/ChatToner2B-2"
HF_TOKEN = os.getenv("HF_TOKEN")

# 서버 설정
HOST = "0.0.0.0"
PORT = 8010

# 추론 설정
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_DO_SAMPLE = True