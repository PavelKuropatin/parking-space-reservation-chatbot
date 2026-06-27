from chatbot.assistant import ParkingAssistant


def main():

    assistant = ParkingAssistant()

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

        out = assistant.chat(text)
        print(f"bot> {out['messages'][-1].content}")


if __name__ == "__main__":
    main()
