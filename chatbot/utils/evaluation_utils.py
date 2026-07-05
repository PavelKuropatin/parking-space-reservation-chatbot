from dataclasses import dataclass
from typing import Literal

from langchain.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from chatbot.database.retriever import ParkingInformationRetriever
from chatbot.prompts import RAG_SUMMARIZATION_PROMPT_TMPL

from chatbot.utils.graph_utils import get_llm
from chatbot.scripts.ingest_static_data import load_static_data
from chatbot.settings import get_settings


@dataclass
class EvaluationDatasetItem:
    question: str
    answer: str
    relevant_categories: list[str]


@dataclass
class EvaluationParams:
    chunk_size: int
    chunk_overlap: int
    top_k: int


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    hits_at_k: float
    mrr: float


class MetricScore(BaseModel):
    score: Literal[0, 0.5, 1]
    reasoning: str


@dataclass
class LLMAnswerMetrics:
    faithfulness: float
    answer_correctness: float


@dataclass
class EvaluationStatus:
    params: EvaluationParams
    retrieval_metrics: RetrievalMetrics
    llm_metrics: LLMAnswerMetrics


# RAG
def calculate_retrieval_metrics(
    items: list[EvaluationDatasetItem],
    retrieved_categories_per_item: list[list[str]],
) -> RetrievalMetrics:
    recalls, precisions, hits, reversed_ranks = [], [], [], []
    for item, received in zip(items, retrieved_categories_per_item):
        relevant = set(item.relevant_categories)
        k = len(received)

        retrieved_set = set(received)
        recalls.append(len(retrieved_set & relevant) / len(relevant))
        precisions.append(sum(1 for c in received if c in relevant) / k if k else 0.0)
        hits.append(1.0 if retrieved_set & relevant else 0.0)

        reversed_rank = 0.0
        for rank, cat in enumerate(received, start=1):
            if cat in relevant:
                reversed_rank = 1.0 / rank
                break
        reversed_ranks.append(reversed_rank)

    n = len(items)
    return RetrievalMetrics(
        recall_at_k=sum(recalls) / n,
        precision_at_k=sum(precisions) / n,
        hits_at_k=sum(hits) / n,
        mrr=sum(reversed_ranks) / n,
    )


# LLM
FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a grader evaluating whether a generated answer is faithful to the
retrieved context — every claim must be directly supported by the context
with no hallucinated information.

Assign a score:
- 1   : Fully faithful. All claims are directly supported by the context.
        Nothing is added from outside the context.
- 0.5 : Partially faithful. Most claims are grounded in the context, but
        at least one claim is not directly supported or is a minor extrapolation.
- 0   : Not faithful. The answer contains significant claims that are absent
        from or contradict the context.
                  """),
        ("human", """
Context:
{rag_context}

Generated answer:
{answer}
                  """),
    ]
)

CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system","""
You are a grader evaluating whether a generated answer is factually correct
compared to a reference answer.
Assign a score:
- 1   : Fully correct. The answer captures all key facts from the reference and introduces no errors.
- 0.5 : Partially correct. The answer captures some key facts but misses important information or contains minor inaccuracies.
- 0   : Incorrect. The answer contradicts the reference, is irrelevant, or fails to address the question.
                  """),
        ("human", """
Reference answer:
{reference}

Generated answer:
{answer}
                  """),
    ]
)


def calculate_faithfulness(
    rag_context: list[str], answer: str, judge_llm: BaseChatModel
) -> float:
    messages = FAITHFULNESS_PROMPT.invoke(
        {"rag_context": rag_context, "answer": answer}
    )
    response = judge_llm.with_structured_output(MetricScore).invoke(messages)
    return response.score


def calculate_correctness(
    reference: str, answer: str, judge_llm: BaseChatModel
) -> float:
    messages = CORRECTNESS_PROMPT.invoke({"reference": reference, "answer": answer})
    response = judge_llm.with_structured_output(MetricScore).invoke(messages)
    return response.score


def evaluate_llm_answers(
    evaluation_dataset: list[EvaluationDatasetItem],
    contexts_per_item: list[list[str]],
    generated_answers: list[str],
    judge_llm: BaseChatModel,
) -> LLMAnswerMetrics:
    faithfulness_s, correctness_s = [], []
    for item, rag_context, answer in zip(
        evaluation_dataset, contexts_per_item, generated_answers
    ):
        faithfulness_s.append(calculate_faithfulness(rag_context, answer, judge_llm))
        correctness_s.append(calculate_correctness(item.answer, answer, judge_llm))
    n = len(evaluation_dataset)
    return LLMAnswerMetrics(
        faithfulness=sum(faithfulness_s) / n,
        answer_correctness=sum(correctness_s) / n,
    )


def generate_llm_answers(
    evaluation_dataset: list[EvaluationDatasetItem],
    rag_context_per_item: list[list[str]],
    llm: BaseChatModel,
) -> list[str]:

    results = []
    for item, rag_context in zip(evaluation_dataset, rag_context_per_item):
        messages = RAG_SUMMARIZATION_PROMPT_TMPL.invoke(
            {
                "rag_context": rag_context,
                "pricing": "",
                "working_hours": "",
                "question": item.question,
            }
        )
        response = llm.invoke(messages)
        results.append(response.content)
    return results


def run_evaluation(
    evaluation_dataset: list[EvaluationDatasetItem],
    retriever: ParkingInformationRetriever,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    llm: BaseChatModel,
    judge_llm: BaseChatModel,
) -> EvaluationStatus:
    retrieved_categories_per_item: list[list[str]] = []
    rag_contexts_per_item: list[list[str]] = []

    for item in evaluation_dataset:
        response = retriever.query(query=item.question, top_k=top_k)
        retrieved_categories_per_item.append(
            [obj.properties.get("category", "") for obj in response.objects]
        )
        rag_contexts_per_item.append(
            [obj.properties.get("content", "") for obj in response.objects]
        )

    retrieval_metrics = calculate_retrieval_metrics(
        evaluation_dataset, retrieved_categories_per_item
    )

    generated_answers = generate_llm_answers(
        evaluation_dataset, rag_contexts_per_item, llm
    )
    llm_metrics = evaluate_llm_answers(
        evaluation_dataset, rag_contexts_per_item, generated_answers, judge_llm
    )

    return EvaluationStatus(
        params=EvaluationParams(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k
        ),
        retrieval_metrics=retrieval_metrics,
        llm_metrics=llm_metrics,
    )


def run_evaluations(
    evaluation_dataset: list[EvaluationDatasetItem],
    collection: str,
    chunk_sizes: list[int],
    chunk_overlaps: list[int],
    top_k_values: list[int] | None = None,
) -> list[EvaluationStatus]:

    settings = get_settings().model_copy(
        update={
            "weaviate_collection": collection,
        }
    )
    llm = get_llm()
    evaluation_results = []
    evaluations_total = len(chunk_sizes) * len(chunk_overlaps)
    evaluation_no = 0

    with ParkingInformationRetriever(settings) as retriever:

        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                evaluation_no += 1
                print(
                    f"[evaluations {evaluation_no} of {evaluations_total}] "
                    f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
                )

                settings = settings.model_copy(
                    update={
                        "rag_chunk_size": chunk_size,
                        "rag_chunk_overlap": chunk_overlap,
                    }
                )
                load_static_data(settings)

                for top_k in top_k_values:
                    print(f"  top_k={top_k}")
                    result = run_evaluation(
                        evaluation_dataset=evaluation_dataset,
                        retriever=retriever,
                        top_k=top_k,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        llm=llm,
                        judge_llm=llm,
                    )
                    evaluation_results.append(result)

    return evaluation_results
