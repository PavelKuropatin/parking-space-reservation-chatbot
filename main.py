from chatbot.database.retriever import ParkingInformationRetriever
from chatbot.utils.settings import get_settings

settings = get_settings()

with ParkingInformationRetriever(settings) as r:

    result = r.query(query="working hours")
    for v in result:
        print(result)
