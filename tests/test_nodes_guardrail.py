from langchain.messages import AIMessage, HumanMessage, RemoveMessage

from chatbot.nodes import (
    input_guardrail_node,
    output_guardrail_node,
    blocked_response_node,
)
from chatbot.prompts import GUARDRAIL_INPUT_BLOCK_MSG, GUARDRAIL_OUTPUT_BLOCK_MSG


# input_guardrail_node
def test_input_guardrail_node_skips_scan_when_no_human_message():
    result = input_guardrail_node({"human_message": ""})
    assert result == {"input_blocked": False, "block_reason": ""}


def test_input_guardrail_node_passes_when_clean():
    result = input_guardrail_node({"human_message": "book me a spot"})

    assert result["input_blocked"] is False
    assert result["block_reason"] == ""
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "book me a spot"


def test_input_guardrail_node_blocks_when_sensitive_data_detected():
    result = input_guardrail_node(
        {"human_message": "card 4111111111111111, iban GB33BUKB20201555555555"}
    )

    assert result == {"input_blocked": True, "block_reason": "CREDIT_CARD, IBAN_CODE"}


# blocked_response_node
def test_blocked_response_node_formats_message():
    result = blocked_response_node({"block_reason": "PHONE_NUMBER"})

    assert len(result["messages"]) == 1
    message = result["messages"][0]
    assert isinstance(message, AIMessage)
    assert message.content == GUARDRAIL_INPUT_BLOCK_MSG.format(details="PHONE_NUMBER")


# output_guardrail_node
def test_output_guardrail_node_returns_empty_when_no_ai_message():
    result = output_guardrail_node({"messages": [HumanMessage("hi")]})

    assert result == {}


def test_output_guardrail_node_passes_when_clean():
    ai_message = AIMessage("your reservation is confirmed")

    result = output_guardrail_node({"messages": [ai_message]})

    assert result == {"human_message": None}


def test_output_guardrail_node_blocks_and_removes_sensitive_ai_message():
    ai_message = AIMessage("your card number is 4111111111111111")

    result = output_guardrail_node({"messages": [ai_message]})

    assert result["human_message"] is None
    messages = result["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], RemoveMessage)
    assert messages[0].id == ai_message.id
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == GUARDRAIL_OUTPUT_BLOCK_MSG.format(
        details="CREDIT_CARD"
    )


def test_output_guardrail_node_picks_last_ai_message():
    first_ai = AIMessage("first")
    human = HumanMessage("in between")
    last_ai_msg = AIMessage("second, the real last one")

    result = output_guardrail_node({"messages": [first_ai, human, last_ai_msg]})

    assert set(result.keys()) == {"human_message"}
    assert result["human_message"] is None
