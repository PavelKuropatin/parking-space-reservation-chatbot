import asyncio

from chatbot.utils.graph_utils import get_checkpointer
from chatbot.settings import get_settings


async def main():

    settings = get_settings()
    async with get_checkpointer(settings) as checkpointer:
        await checkpointer.setup()


if __name__ == "__main__":
    asyncio.run(main())
