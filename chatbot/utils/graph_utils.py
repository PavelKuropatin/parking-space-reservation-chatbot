from datetime import datetime

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from chatbot.settings import Settings, get_settings


def get_llm() -> BaseChatModel:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=settings.openai_llm_api_key,
        base_url=settings.openai_llm_url,
    )


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def get_checkpointer(settings: Settings) -> PostgresSaver:
    try:

        url = "postgresql://{user}:{pswd}@{host}/{db}".format(  # pylint: disable=consider-using-f-string
            user=settings.checkpointer_user,
            pswd=settings.checkpointer_pswd,
            host=settings.checkpointer_host,
            db=settings.checkpointer_db,
        )
        return PostgresSaver.from_conn_string(url)

    except Exception as e:
        raise e


def last_ai(messages: list[BaseMessage]) -> AIMessage:
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None
