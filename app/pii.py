from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
import re

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()


def redact_pii(text):
    results = analyzer.analyze(text=text, language="en")
    if not results:
        return text
    redacted = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
            "US_BANK_NUMBER": OperatorConfig("replace", {"new_value": "[ID_NUMBER]"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "[PHONE]"}),
        },
    )
    return redacted.text


def redact_pii_indonesian(text):
    # NIK (16 digit)
    text = re.sub(r"\b\d{16}\b", "[NIK]", text)
    # Nomor telepon Indonesia
    text = re.sub(
        r"\b(\+62|62|0)[\s-]?\d{2,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b", "[PHONE]", text
    )
    # Email
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL]", text)
    return text


def detect_and_redact(text):
    text = redact_pii_indonesian(text)
    text = redact_pii(text)
    return text
