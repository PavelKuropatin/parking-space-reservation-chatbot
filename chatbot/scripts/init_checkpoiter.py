from chatbot.utils.graph_utils import get_checkpointer
from chatbot.settings import get_settings


def main():

    settings = get_settings()
    with get_checkpointer(settings) as checkpointer:
        checkpointer.setup()


if __name__ == "__main__":
    main()
