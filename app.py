from uuid import uuid4

from chatbot.assistant import ParkingAssistant
from chatbot.database.retriever import get_parking_info_retriever
from chatbot.database.sql_store import get_parking_data_db


def main():

    thread_id = str(uuid4())
    assistant = ParkingAssistant(thread_id)

    with (
        get_parking_data_db() as _,
        get_parking_info_retriever() as _
    ):
        while True:
            try:
                text = input("\n\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue

            state = assistant.chat(text)
            print(f"bot> {state['messages'][-1].content}")


if __name__ == "__main__":
    main()
