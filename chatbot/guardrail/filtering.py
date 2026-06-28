from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

OUTPUT_BLOCKED_ENTITIES = [
    "IBAN_CODE",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
]
INPUT_BLOCKED_ENTITIES = OUTPUT_BLOCKED_ENTITIES + ["PHONE_NUMBER"]
SCORE_THRESHOLD = 0.5


@dataclass
class DetectionResult:
    blocked: bool
    details: list[RecognizerResult] = field(default_factory=list)

    @property
    def entity_names(self) -> list[str]:
        return sorted({d.entity_type for d in self.details})

    @property
    def entity_summary(self) -> str:
        return ", ".join(self.entity_names)


class Guardtrail:

    def __init__(self):
        self.__analyzer = self.__build_analyzer()

    def __build_analyzer(self) -> AnalyzerEngine:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            }
        )
        nlp_engine = provider.create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    def for_input(self, text: str) -> DetectionResult:
        return self.__scan(text, INPUT_BLOCKED_ENTITIES)

    def for_output(self, text: str) -> DetectionResult:
        return self.__scan(text, OUTPUT_BLOCKED_ENTITIES)

    def __scan(self, text: str, blocked_entities: list[str]) -> DetectionResult:
        if not text or not text.strip():
            return DetectionResult(blocked=False)

        results = self.__analyzer.analyze(
            text=text,
            entities=blocked_entities,
            language="en",
            score_threshold=SCORE_THRESHOLD,
        )

        if results:
            return DetectionResult(blocked=True, details=results)
        return DetectionResult(blocked=False)


__GUARDRAIL = None


def get_guardrail() -> Guardtrail:
    global __GUARDRAIL
    if __GUARDRAIL is None:
        __GUARDRAIL = Guardtrail()
    return __GUARDRAIL
