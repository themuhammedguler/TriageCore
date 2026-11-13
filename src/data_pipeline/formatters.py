"""Formatters for transforming cleaned medical dialogues into standard ChatML / Llama-3 instruction datasets.

Ensures proper system prompt injection, role alignment, and JSONL persistence.
"""

import json
from pathlib import Path
from typing import Iterable, Union

from src.data_pipeline.schemas import (
    ChatMessage,
    MessageRole,
    RawMedicalRecord,
    TriageSample,
)

DEFAULT_TRIAGE_SYSTEM_PROMPT = (
    "You are TriageCore, an expert AI clinical triage assistant. "
    "Your objective is to evaluate patient symptoms, assess medical urgency, "
    "highlight critical red-flag warnings, explain potential etiologies, and recommend appropriate clinical next steps.\n"
    "Always advise immediate emergency medical care (call 112/911 or proceed to the nearest emergency department) "
    "for severe symptoms such as acute chest pain, shortness of breath, sudden numbness or speech difficulty, or severe uncontrolled bleeding. "
    "TriageCore provides clinical decision support and educational triage; it does not replace licensed in-person medical diagnosis."
)


def format_to_triage_sample(
    record: RawMedicalRecord,
    system_prompt: str = DEFAULT_TRIAGE_SYSTEM_PROMPT,
) -> TriageSample:
    """Transform a cleaned raw medical record into a validated 3-turn ChatML TriageSample."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=record.patient_query),
        ChatMessage(role=MessageRole.ASSISTANT, content=record.doctor_response),
    ]
    return TriageSample(messages=messages)


def sample_to_dict(sample: TriageSample) -> dict:
    """Serialize a TriageSample to a dictionary conforming to HuggingFace / Llama-3 standard format."""
    return {
        "messages": [
            {"role": msg.role.value, "content": msg.content}
            for msg in sample.messages
        ]
    }


def save_samples_to_jsonl(
    samples: Iterable[TriageSample],
    output_file: Union[str, Path],
) -> int:
    """Save an iterable of TriageSample objects into a JSON Lines (.jsonl) file.

    Returns:
        int: Number of samples written.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            payload = sample_to_dict(sample)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1

    return count
