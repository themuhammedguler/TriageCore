"""Unit tests for Phase 1: Data Curation & Cleaning Pipeline."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data_pipeline.clean import (
    anonymize_pii,
    process_record,
    sanitize_text,
    validate_quality,
)
from src.data_pipeline.formatters import (
    DEFAULT_TRIAGE_SYSTEM_PROMPT,
    format_to_triage_sample,
    sample_to_dict,
    save_samples_to_jsonl,
)
from src.data_pipeline.schemas import (
    ChatMessage,
    MessageRole,
    RawMedicalRecord,
    TriageSample,
)


def test_schema_valid_sample():
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
        ChatMessage(role=MessageRole.USER, content="I have a severe headache."),
        ChatMessage(role=MessageRole.ASSISTANT, content="Assessment: Evaluate for red flags."),
    ]
    sample = TriageSample(messages=messages)
    assert len(sample.messages) == 3
    assert sample.messages[1].role == MessageRole.USER


def test_schema_rejects_empty_content():
    with pytest.raises(ValidationError):
        ChatMessage(role=MessageRole.USER, content="   ")


def test_schema_rejects_missing_roles():
    with pytest.raises(ValidationError):
        # Only user message, no assistant
        TriageSample(messages=[ChatMessage(role=MessageRole.USER, content="Hello")])


def test_pii_anonymization():
    raw_text = (
        "Hello, I am Dr. Johnathan Smith. You can reach me at doctor.smith@hospital.org "
        "or call +1 (555) 123-4567. Visit https://clinic.com for appointments. Regards, Dr. Smith."
    )
    cleaned = anonymize_pii(raw_text)
    assert "doctor.smith@hospital.org" not in cleaned
    assert "[EMAIL]" in cleaned
    assert "+1 (555) 123-4567" not in cleaned
    assert "[PHONE]" in cleaned
    assert "https://clinic.com" not in cleaned
    assert "[LINK]" in cleaned
    assert "Dr. Smith" not in cleaned


def test_sanitize_text():
    dirty = "Line 1\r\nLine 2\x00\x08   with    spaces\n\n\n\nLine 3"
    sanitized = sanitize_text(dirty)
    assert "\x00" not in sanitized
    assert "\r\n" not in sanitized
    assert "with spaces" in sanitized
    assert "\n\n\n\n" not in sanitized


def test_validate_quality_thresholds():
    # Too short query
    valid, reason = validate_quality("help", "This is a detailed response from the clinician explaining symptoms.")
    assert not valid
    assert "Patient query too short" in reason

    # Too short response
    valid, reason = validate_quality("I have been having chronic abdominal pain for 3 weeks.", "See a doctor.")
    assert not valid

    # Generic discard
    valid, reason = validate_quality(
        "I have a fever of 39 C and severe fatigue.",
        "Please consult a doctor."
    )
    assert not valid
    assert "Generic non-informative" in reason

    # Valid case
    valid, reason = validate_quality(
        "Patient presents with sudden unilateral chest tightness radiating to the left arm.",
        "EMERGENCY: Immediate evaluation for acute coronary syndrome required. Call emergency services."
    )
    assert valid
    assert reason is None


def test_process_record_pipeline():
    record = RawMedicalRecord(
        patient_query="   My left knee is swollen and hot to the touch after playing soccer.   ",
        doctor_response="Regards, Dr. Alice. Clinical assessment: Evaluate for acute joint effusion or septic arthritis.",
        source="test_source",
    )
    processed = process_record(record)
    assert processed is not None
    assert "Dr. Alice" not in processed.doctor_response
    assert processed.patient_query.startswith("My left knee")


def test_format_and_save_jsonl(tmp_path: Path):
    record = RawMedicalRecord(
        patient_query="Patient exhibits mild fever and productive cough.",
        doctor_response="Recommend rest, adequate oral hydration, and monitoring oxygen saturation.",
        source="test",
    )
    sample = format_to_triage_sample(record)
    assert len(sample.messages) == 3
    assert sample.messages[0].content == DEFAULT_TRIAGE_SYSTEM_PROMPT

    dict_repr = sample_to_dict(sample)
    assert "messages" in dict_repr
    assert dict_repr["messages"][0]["role"] == "system"

    output_file = tmp_path / "test_out.jsonl"
    count = save_samples_to_jsonl([sample], output_file)
    assert count == 1
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        line = f.readline()
        loaded = json.loads(line)
        assert len(loaded["messages"]) == 3


def test_golden_triage_turkish_dataset():
    """Verify that the standalone Golden Triage Turkish dataset exists, is valid JSONL,

    contains at least 300 rich clinical cases across all 5 ESI levels, and follows ChatML format.
    """
    dataset_path = Path("data/processed/golden_triage_turkish.jsonl")
    assert dataset_path.exists(), f"Golden triage dataset not found at {dataset_path}"

    cases = []
    esi_levels_found = set()

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            assert "messages" in data, f"Line {line_no} missing 'messages' key"
            messages = data["messages"]
            assert len(messages) >= 3, f"Line {line_no} has fewer than 3 messages"

            roles = [m["role"] for m in messages]
            assert roles[0] == "system"
            assert roles[1] == "user"
            assert roles[2] == "assistant"

            user_text = messages[1]["content"]
            assistant_text = messages[2]["content"]
            assert len(user_text) > 10, f"Line {line_no} user prompt too short"
            assert len(assistant_text) > 50, f"Line {line_no} assistant prompt too short"

            # Check for Turkish clinical structure indicators
            assert any(tag in assistant_text for tag in ["Klinik", "ESI", "Aciliyet", "Protokol", "Eylem"]), (
                f"Line {line_no} assistant response missing clinical structure"
            )

            # Check ESI level coverage from text or metadata
            meta_level = data.get("metadata", {}).get("esi_level")
            if meta_level:
                esi_levels_found.add(meta_level)
            for level in range(1, 6):
                if f"ESI Seviye {level}" in assistant_text or f"ESI-{level}" in assistant_text or f"Seviye: {level}" in assistant_text:
                    esi_levels_found.add(level)

            cases.append(data)

    assert len(cases) >= 300, f"Expected at least 300 golden cases, got {len(cases)}"
    assert esi_levels_found == {1, 2, 3, 4, 5}, f"Missing some ESI levels in golden dataset: {esi_levels_found}"

