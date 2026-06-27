from langchain.messages import AIMessage, AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI

from chatbot.settings import get_settings


def get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_llm_model,
        api_key=settings.openai_llm_api_key,
        base_url=settings.openai_llm_url,
    )


def last_user_input(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def last_ai_output(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content
    return ""
