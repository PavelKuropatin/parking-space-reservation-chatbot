import pytest

from chatbot.guardrail.filtering import (
    Guardrail,
)


# --------------------------------------------------------------------------- #
# Guardtrail
# --------------------------------------------------------------------------- #
@pytest.fixture
def guardrail():
    return Guardrail()


def test_for_input_returns_not_blocked_on_empty_text(guardrail):
    result = guardrail.for_input("")

    assert result.blocked is False


def test_for_input_returns_not_blocked_on_whitespace_text(guardrail):
    result = guardrail.for_input("   \n\t  ")

    assert result.blocked is False


def test_for_input_not_blocked_when_no_entities_detected(guardrail):
    result = guardrail.for_input("hello there")

    assert result.blocked is False
    assert result.details == []


def test_for_input_blocks_when_entities_detected(guardrail):
    result = guardrail.for_input("card 4111111111111111, iban GB33BUKB20201555555555")

    assert result.blocked is True
    assert result.entity_names == ["CREDIT_CARD", "IBAN_CODE"]


def test_for_output_blocks_when_entities_detected(guardrail):
    result = guardrail.for_output("card 4111111111111111, iban GB33BUKB20201555555555")

    assert result.blocked is True
    assert result.entity_names == ["CREDIT_CARD", "IBAN_CODE"]


def test_for_output_returns_not_blocked_on_empty_text(guardrail):
    result = guardrail.for_output("")

    assert result.blocked is False
