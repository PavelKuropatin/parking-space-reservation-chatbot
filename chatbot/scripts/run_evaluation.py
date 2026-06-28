from chatbot.utils.evaluation_utils import (
    EvaluationDatasetItem,
    EvaluationStatus,
    run_evaluations,
)

EVALUATION_DATASET: list[EvaluationDatasetItem] = [
    # general
    EvaluationDatasetItem(
        question="Who operates CityPark?",
        answer="CityPark is the operator.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="How far in advance can I reserve a parking space?",
        answer="You can reserve from 30 minutes up to 90 days in advance.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="What payment methods does CityPark accept?",
        answer="Credit/debit card, Apple Pay, Google Pay, and in-app wallet.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="What is the address of CityPark Central?",
        answer="12 Central Avenue, Riga, LV-1010, Latvia.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="What are the height limits at CityPark Central?",
        answer="2.10 m on Levels 1-2 and 2.50 m on Level 3.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="How do I get to CityPark Central by public transport?",
        answer="Trams 3, 7, 9 and buses 22, 40 stop at Central Station, a 4-minute walk from the facility.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="What parking space types are available at CityPark?",
        answer="Standard, Compact, Oversized/SUV, EV Charging, and Accessible.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="Is EV charging energy included in the parking rate?",
        answer="No, energy for EV charging is billed separately from the parking rate.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="What is the CityPark customer support phone number?",
        answer="+371 6000 0000, available 24/7.",
        relevant_categories=["general"],
    ),
    EvaluationDatasetItem(
        question="What are the hours for CityPark in-app chat support?",
        answer="In-app chat is available 06:00-24:00.",
        relevant_categories=["general"],
    ),
    # booking
    EvaluationDatasetItem(
        question="How do I make a parking reservation at CityPark?",
        answer=(
            "Open the CityPark app or website, enter your location, arrival date, time, and duration, "
            "choose a space type, provide your licence plate number, and pay. "
            "You will receive a confirmation and QR code by email and in the app."
        ),
        relevant_categories=["booking"],
    ),
    EvaluationDatasetItem(
        question="How do I enter the parking facility after booking?",
        answer=(
            "At the barrier the camera reads your licence plate automatically. "
            "If recognition fails, scan your QR code at the reader."
        ),
        relevant_categories=["booking"],
    ),
    EvaluationDatasetItem(
        question="Can I extend my reservation after it starts?",
        answer="Yes, you can add time from the app, subject to availability.",
        relevant_categories=["booking"],
    ),
    EvaluationDatasetItem(
        question="How do I modify my reservation date or time?",
        answer="You can change the date/time up to 1 hour before the start time from the app; a price difference may apply.",
        relevant_categories=["booking"],
    ),
    EvaluationDatasetItem(
        question="How long does a refund take after cancelling a reservation?",
        answer="A full refund is returned to the original payment method within 3-5 business days.",
        relevant_categories=["booking"],
    ),
    # policy
    EvaluationDatasetItem(
        question="Can I use my reservation at a different CityPark location than booked?",
        answer="No, a reservation is valid only for the booked location, time window, and space type.",
        relevant_categories=["policies"],
    ),
    EvaluationDatasetItem(
        question="Can I park a different vehicle than the one I booked with?",
        answer="No, one reservation covers one vehicle and the licence plate must match the booking.",
        relevant_categories=["policies"],
    ),
    EvaluationDatasetItem(
        question="What activities are prohibited inside CityPark facilities?",
        answer="Overnight sleeping, vehicle repairs, and commercial activity are not allowed on-site.",
        relevant_categories=["policies"],
    ),
    EvaluationDatasetItem(
        question="Is CityPark liable if my vehicle is stolen or damaged?",
        answer="No, the facility is not liable for theft or damage. You should lock your vehicle and park at your own risk.",
        relevant_categories=["policies"],
    ),
    EvaluationDatasetItem(
        question="What happens if I stay longer than my booked time window?",
        answer="Overstaying converts to drive-up pricing for the extra time.",
        relevant_categories=["policies", "booking"],
    ),
    # qa
    EvaluationDatasetItem(
        question="Do I need to print my reservation confirmation to enter?",
        answer="No, entry is automatic via licence-plate recognition. The QR code in the app is a backup.",
        relevant_categories=["faq"],
    ),
    EvaluationDatasetItem(
        question="Can I update my licence plate number after booking?",
        answer="Yes, you can edit the plate in the app up to 1 hour before arrival.",
        relevant_categories=["faq"],
    ),
    EvaluationDatasetItem(
        question="Can I enter the parking facility before my booked start time?",
        answer="Yes, you can enter up to 15 minutes before your start time at no charge.",
        relevant_categories=["faq"],
    ),
    EvaluationDatasetItem(
        question="What should I do if the barrier camera does not recognize my plate?",
        answer="Press the help button at the barrier or scan your QR code. Support can open the gate remotely 24/7.",
        relevant_categories=["faq"],
    ),
    EvaluationDatasetItem(
        question="Can I reserve an EV charger without booking a parking space?",
        answer="No, EV bays include both the parking space and the charger. Select an EV Charging space when booking.",
        relevant_categories=["faq", "general"],
    ),
    EvaluationDatasetItem(
        question="Is cancellation free and when does it stop being free?",
        answer="Cancellation is free up to 1 hour before your start time. Later cancellations are non-refundable.",
        relevant_categories=["faq", "booking"],
    ),
    EvaluationDatasetItem(
        question="How do I get a VAT invoice for my parking?",
        answer="Receipts are emailed automatically. You can download a VAT invoice anytime from My Bookings in the app.",
        relevant_categories=["faq"],
    ),
    EvaluationDatasetItem(
        question="Are vans or campervans allowed at CityPark?",
        answer="Yes, where an Oversized/SUV bay is available and within the site's height limit.",
        relevant_categories=["faq", "policies"],
    ),
    EvaluationDatasetItem(
        question="What happens if no space is available when I arrive with a confirmed reservation?",
        answer=(
            "A confirmed reservation guarantees a space. "
            "If one is somehow unavailable, staff will direct you to an alternative bay "
            "or a nearby partner facility at no extra cost."
        ),
        relevant_categories=["faq", "policies"],
    ),
    EvaluationDatasetItem(
        question="What requirements are needed to park in an accessible bay?",
        answer="A valid disabled-parking permit must be displayed on the dashboard.",
        relevant_categories=["policies", "general"],
    ),
]


def print_results(results: list[EvaluationStatus]) -> None:
    for r in results:
        print(f"""
---------------------------------------------
Params
    Chunk size   :{r.params.chunk_size}
    Chunk overlap:{r.params.chunk_overlap}
    K            :{r.params.top_k}
Retrieval metrics
    Recall@K     :{r.retrieval_metrics.recall_at_k:.3f}
    Precition@K  :{r.retrieval_metrics.precision_at_k:.3f}
    Hit@K        :{r.retrieval_metrics.hits_at_k:.3f}
    MRR          :{r.retrieval_metrics.mrr:.3f}
LLM Answer/Quality metrics:
    Fiathfullness:{r.llm_metrics.faithfulness:.3f}
    Correctness  :{r.llm_metrics.answer_correctness:.3f}
---------------------------------------------
""")

def main():
    results = run_evaluations(
        evaluation_dataset=EVALUATION_DATASET,
        collection="evaluation_collection",
        chunk_sizes=[200],
        chunk_overlaps=[100],
        top_k_values=[3],
    )
    print_results(results)

if __name__ == "__main__":
    main()
