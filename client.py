import asyncio
import sys
import time
from typing import Optional
from uuid import uuid4

from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import StateSnapshot
from langgraph.graph.state import Command, CompiledStateGraph
from chatbot.database.retriever import get_parking_info_retriever
from chatbot.database.sql_store import get_parking_data_db
from chatbot.graph import build_graph
from chatbot.notifier import Notifier, RequestType, get_notifier
from chatbot.states import GraphState
from chatbot.utils.graph_utils import get_checkpointer
from chatbot.settings import get_settings


async def print_history(
    graph: CompiledStateGraph, config: RunnableConfig, n: int = 5
) -> None:
    state_snapshot = await graph.aget_state(config)
    messages = (state_snapshot.values or {}).get("messages", [])

    if messages:
        print("[HISTORY_START]", "-" * 22)
        for message in messages[-n:]:
            if isinstance(message, HumanMessage):
                print("you>", message.content)
            if isinstance(message, AIMessage):
                print("bot>", message.content)
        print("[HISTORY_END]", "-" * 22)


def get_interrupt_payload(obj: GraphState | StateSnapshot) -> Optional[dict]:
    if not obj:
        return None

    if isinstance(obj, StateSnapshot):
        interrupts = obj.interrupts
    else:
        interrupts = obj.get("__interrupt__", [])

    if interrupts:
        return interrupts[0].value
    return None


async def chat(
    graph: CompiledStateGraph, config: RunnableConfig, notifier: Notifier
) -> None:

    while True:
        try:
            text = input("\n\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in {"quit", "exit"}:
            break

        if not text:
            continue

        # chatbot
        state = await graph.ainvoke({"human_message": text}, config=config)
        if interrupt_payload := get_interrupt_payload(state):
            await await_and_resume(graph, config, notifier, interrupt_payload)
        else:
            print(f"bot> {state['messages'][-1].content}")


async def await_and_resume(
    graph: GraphState, config: dict, notifier: Notifier, payload: dict
) -> None:

    while payload is not None:
        request_id = payload["request_id"]
        notifier.notify(request_id, RequestType.REQUEST, payload)

        decision = None
        while not decision:
            decision = notifier.pull(request_id, RequestType.RESPONSE)
            print(f"system> Waiting for admin decision for request {request_id}...")
            time.sleep(5)

        state = await graph.ainvoke(Command(resume=decision), config)
        payload = get_interrupt_payload(state)
        if payload is None:
            print(f"bot> {state['messages'][-1].content}")


async def main():

    args = sys.argv[1:]
    if len(args):
        client_id = args[0]
    else:
        client_id = f"client-id-{str(uuid4())}"
    config: RunnableConfig = {"configurable": {"thread_id": client_id}}
    print(f"Using thread_id: {config['configurable']['thread_id']}")

    settings = get_settings()
    notifier = get_notifier()

    async with get_checkpointer(settings) as checkpointer:
        with get_parking_data_db() as _, get_parking_info_retriever() as _:

            client_graph = await build_graph(checkpointer)
            await print_history(client_graph, config)

            state = await client_graph.aget_state(config)
            interrupt_payload = get_interrupt_payload(state)
            if interrupt_payload:
                await await_and_resume(
                    client_graph, config, notifier, interrupt_payload
                )
            await chat(client_graph, config, notifier)


if __name__ == "__main__":
    asyncio.run(main())
