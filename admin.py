import time

from langgraph.types import Command
from chatbot.graph import build_graph
from chatbot.notifier import RequestType, get_notifier
from chatbot.states import ReservationStatus
from chatbot.utils.graph_utils import get_checkpointer
from chatbot.logging import logger
from chatbot.settings import get_settings


def main():

    settings = get_settings()
    _ = {"configurable": {"thread_id": "admin-id-9996c969-8e5c-484f-af07-37683fa07e80"}}

    with get_checkpointer(settings) as checkpointer:
        admin_graph = build_graph(checkpointer)
        notifier = get_notifier()
        while True:
            print("Waiting for admin notification...")
            if payloads := notifier.pull_all(RequestType.REQUEST, n=1):

                try:
                    for payload in payloads:
                        print("\n\n=== Admin Notification START ===")

                        # info
                        request_id = payload["request_id"]
                        reservation_id = payload["reservation_id"]
                        calling_thread_id = payload["calling_thread_id"]
                        message = payload["message"]
                        print(
                            (
                                "Notification received:\n"
                                f" - Request ID: {request_id}\n"
                                f" - Calling Thread ID: {calling_thread_id}\n"
                                f" - Reservation ID: {reservation_id}\n"
                                f"{message}\n"
                                f"Please approve ({ReservationStatus.APPROVED.value} / {ReservationStatus.REJECTED.value})"
                            )
                        )

                        # wait for response
                        while response := input("Response (a/r): "):
                            response = response.strip().lower()
                            if response in ("a", "r"):
                                break

                        match response:
                            case "a":
                                reservation_status = ReservationStatus.APPROVED
                            case "r":
                                reservation_status = ReservationStatus.REJECTED

                        # response to "client"
                        output_payload = {
                            "calling_thread_id": calling_thread_id,
                            "request_id": request_id,
                            "reservation_id": reservation_id,
                            "decision": reservation_status.value,
                        }
                        notifier.notify(
                            payload["request_id"], RequestType.RESPONSE, output_payload
                        )
                        _ = admin_graph.invoke(
                            Command(resume=reservation_status.value),
                            config={"configurable": {"thread_id": calling_thread_id}},
                        )
                        print("Notification sent.")
                        notifier.remove(payload["request_id"], RequestType.REQUEST)
                        print("=== Admin Notification END ===\n\n")

                except Exception as e:
                    logger.error("Error processing admin notification: %s", e)

            time.sleep(5)


if __name__ == "__main__":
    main()
