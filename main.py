from chatbot.assistant import ParkingAssistant
from chatbot.database.sql_store import ParkingData
from chatbot.settings import get_settings
from chatbot.utils.weaviate_utils import get_weaviate_client, get_weaviate_vector_store


def main():

    # todo simple cli to provide user id ? optionally continue conversation
    settings = get_settings()

    with (
        ParkingData(settings) as parking_data_db,
        get_weaviate_client(settings) as w_client,
    ):
        parking_info_vs = get_weaviate_vector_store(w_client, settings)
        assistant = ParkingAssistant(settings, parking_data_db, parking_info_vs)

        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            finally:
                # todo smth to close all connections
                pass

            if text.lower() in {"quit", "exit"}:
                break
            if not text:
                continue

            chat_state = assistant.chat(text)
            print(f"bot> \n")
            for m in chat_state["messages"]:
                print("---->", m.content)


if __name__ == "__main__":
    main()
