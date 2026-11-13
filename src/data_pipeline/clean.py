"""Data cleaning and sanitization pipeline for raw medical dialogues.

Provides regex-based PII anonymization, length thresholds, quality filters,
and clinical content validation.
"""

import re
from typing import Optional, Tuple

from src.data_pipeline.schemas import RawMedicalRecord

# Common doctor greeting and sign-off patterns to anonymize
DOCTOR_SIGNATURE_PATTERNS = [
    re.compile(r"(regards|sincerely|best regards|warm regards|thanks)[,\s]+(dr\.?|doctor)?\s*[a-zA-Z\s\.]+", re.IGNORECASE),
    re.compile(r"(dr\.?|doctor)\s+[a-z]+(?:\s+[a-z]+)?", re.IGNORECASE),
    re.compile(r"\[Doctor Name\]|\[Hospital Name\]|\[Clinic Name\]", re.IGNORECASE),
]

# PII patterns
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Common web scrape artifacts in online doctor datasets
WEB_ARTIFACT_PATTERNS = [
    re.compile(r"\s*--+>\s*$", re.IGNORECASE),
    re.compile(r"click here to (read|consult|chat).*", re.IGNORECASE),
    re.compile(r"revert back after a week to an? [a-z\s]+ online.*", re.IGNORECASE),
]

# Generic unhelpful responses to filter out
GENERIC_DISCARD_PATTERNS = [
    re.compile(r"^(please )?(consult|visit|see) a (doctor|physician)\.?$", re.IGNORECASE),
    re.compile(r"^(i don\'t know|i cannot help you with this)\.?$", re.IGNORECASE),
    re.compile(r"^hello\s*,?\s*how can i help you\??$", re.IGNORECASE),
    re.compile(r".*(revert back|consult).*online\s*--?>?$", re.IGNORECASE),
]


def sanitize_text(text: str) -> str:
    """Normalize whitespace, remove control characters, web scrap artifacts and non-printable artifacts."""
    if not text:
        return ""
    # Strip null bytes and control chars except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normalize multiple newlines and spaces
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing web scrape artifacts
    for pattern in WEB_ARTIFACT_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def anonymize_pii(text: str) -> str:
    """Mask doctor names, emails, phone numbers, and web links."""
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = URL_PATTERN.sub("[LINK]", text)
    for pattern in DOCTOR_SIGNATURE_PATTERNS:
        text = pattern.sub("Attending Clinician", text)
    return text


def clean_medical_text(text: str) -> str:
    """Apply sanitization and PII anonymization to a single text field."""
    text = sanitize_text(text)
    text = anonymize_pii(text)
    return text


def validate_quality(
    patient_query: str,
    doctor_response: str,
    min_query_len: int = 25,
    min_response_len: int = 50,
    max_total_chars: int = 3500,
) -> Tuple[bool, Optional[str]]:
    """Determine if a pair meets minimum information density and sequence length constraints."""
    # Discard canned / non-informative responses regardless of length
    for pattern in GENERIC_DISCARD_PATTERNS:
        if pattern.match(doctor_response.strip()):
            return False, "Generic non-informative response"

    if len(patient_query) < min_query_len:
        return False, f"Patient query too short ({len(patient_query)} < {min_query_len})"

    if len(doctor_response) < min_response_len:
        return False, f"Doctor response too short ({len(doctor_response)} < {min_response_len})"

    total_chars = len(patient_query) + len(doctor_response)
    if total_chars > max_total_chars:
        return False, f"Total characters exceed maximum limit ({total_chars} > {max_total_chars})"

    return True, None


def process_record(record: RawMedicalRecord) -> Optional[RawMedicalRecord]:
    """Clean and validate a raw medical record. Returns None if record fails quality checks."""
    cleaned_query = clean_medical_text(record.patient_query)
    cleaned_response = clean_medical_text(record.doctor_response)

    is_valid, reason = validate_quality(cleaned_query, cleaned_response)
    if not is_valid:
        return None

    return RawMedicalRecord(
        patient_query=cleaned_query,
        doctor_response=cleaned_response,
        source=record.source,
        metadata=record.metadata,
    )
