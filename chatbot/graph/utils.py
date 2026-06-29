from datetime import datetime

from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from chatbot.settings import get_settings


def get_llm() -> BaseChatModel:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=settings.openai_llm_api_key,
        base_url=settings.openai_llm_url,
    )


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")
