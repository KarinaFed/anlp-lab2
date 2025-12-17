"""Configuration for the multi-agent system"""
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

# vLLM endpoint configuration
BASE_URL = os.getenv("LITELLM_BASE_URL", "http://a6k2.dgx:34000/v1")
API_KEY = os.getenv("LITELLM_API_KEY", "sk-jZkA340PLjFS8B47HeFHsw")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-32b")

# Memory storage path
MEMORY_STORAGE_PATH = os.getenv("MEMORY_STORAGE_PATH", "memory_store.json")

