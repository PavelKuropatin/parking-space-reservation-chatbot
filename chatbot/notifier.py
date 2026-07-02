import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from chatbot.settings import Settings, get_settings


class RequestType(Enum):
    REQUEST = "request"
    RESPONSE = "response"


class Notifier:

    def __init__(self, settings: Settings):
        self.__root = settings.notification_path
        self.__paths = {
            RequestType.REQUEST: os.path.join(self.__root, RequestType.REQUEST.value),
            RequestType.RESPONSE: os.path.join(self.__root, RequestType.RESPONSE.value),
        }

        os.makedirs(self.__root, exist_ok=True)
        for p in self.__paths.values():
            os.makedirs(p, exist_ok=True)

    def notify(self, request_id: str, request_type: RequestType, payload: dict) -> None:
        path = os.path.join(self.__paths[request_type], f"{request_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))

    def pull(self, request_id: str, request_type: RequestType) -> Optional[list[dict]]:
        path = os.path.join(self.__paths[request_type], f"{request_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return json.loads(content)

    def pull_all(self, request_type: RequestType, n: int) -> list[dict]:
        output = []
        for path in Path(self.__paths[request_type]).glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            output.append(json.loads(content))
            if len(output) == n:
                break
        return output

    def remove(self, request_id: str, request_type: RequestType) -> None:
        path = os.path.join(self.__paths[request_type], f"{request_id}.json")
        if os.path.exists(path):
            os.remove(path)


__NOTIFIER: Notifier = None


def get_notifier() -> Notifier:
    global __NOTIFIER
    if __NOTIFIER is None:
        __NOTIFIER = Notifier(get_settings())
    return __NOTIFIER
