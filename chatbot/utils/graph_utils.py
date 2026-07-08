from datetime import datetime

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from chatbot.settings import Settings, get_settings


def get_llm(temperature: float) -> BaseChatModel:
    settings = get_settings()
    match settings.llm_model_provider:
        case "openai":
            return ChatOpenAI(
                model=settings.llm_model_name,
                api_key=settings.llm_api_key,
                base_url=settings.llm_url,
                temperature=temperature,
            )
        case "anthropic":
            return ChatAnthropic(
                model=settings.llm_model_name,
                api_key=settings.llm_api_key,
                temperature=temperature,
            )


def now(fmt: str = "%Y-%m-%d %H:%M"):
    return datetime.now().strftime(fmt)


def get_checkpointer(settings: Settings) -> AsyncPostgresSaver:
    try:

        url = "postgresql://{user}:{pswd}@{host}/{db}".format(  # pylint: disable=consider-using-f-string
            user=settings.checkpointer_user,
            pswd=settings.checkpointer_pswd,
            host=settings.checkpointer_host,
            db=settings.checkpointer_db,
        )
        return AsyncPostgresSaver.from_conn_string(url)

    except Exception as e:
        raise e


def last_ai(messages: list[BaseMessage]) -> AIMessage:
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None

def last_human(messages: list[BaseMessage]) -> HumanMessage:
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            return m
    return None
