#!/usr/bin/env python3
"""
Shared text and conversation formatting utilities for VLM experiments.
"""

import re
import json
from typing import Any, Dict, List
from PIL import Image
from io import BytesIO


def clean_json_output(raw_output: str) -> str:
    """
    Clean model output to extract only the JSON object.
    Removes markdown code blocks, extra text, and formatting.

    Args:
        raw_output: Raw output from the model

    Returns:
        Clean JSON string

    Example:
        >>> raw = '```json\\n{"date": "2024-01-15"}\\n```'
        >>> clean_json_output(raw)
        '{"date": "2024-01-15"}'
    """
    # Remove markdown code blocks
    cleaned = re.sub(r'```json\s*', '', raw_output)
    cleaned = re.sub(r'```\s*$', '', cleaned)

    # Extract JSON object
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')

    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        cleaned = cleaned[start_idx:end_idx + 1]

    return cleaned.strip()


def convert_to_conversation(
    image: Image.Image,
    instruction: str,
    response: str
) -> Dict[str, List[Dict]]:
    """
    Convert a sample to Unsloth chat format for training.

    Args:
        image: PIL Image object
        instruction: User instruction/prompt text
        response: Expected assistant response (for training)

    Returns:
        Dict with 'messages' key containing conversation

    Example:
        >>> conv = convert_to_conversation(pil_image, "Extract text", '{"text": "hello"}')
        >>> # Returns: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": response}
            ]
        },
    ]
    return {"messages": conversation}


def convert_kie_sample_to_conversation(
    sample: Dict[str, Any],
    prompt: str
) -> Dict[str, List[Dict]]:
    """
    Convert a KIE dataset sample to conversation format.
    Handles the specific structure of KIE datasets (image bytes, annotations dict).

    Args:
        sample: Dataset sample with 'image' and 'annotations' fields
        prompt: Instruction prompt for the model

    Returns:
        Dict with 'messages' key containing conversation
    """
    # Decode bytes to PIL Image if needed
    if isinstance(sample["image"], bytes):
        image = Image.open(BytesIO(sample["image"]))
    else:
        image = sample["image"]

    # Get annotations (handle different field names)
    annotations = sample.get(
        "annotations",
        sample.get("ground_truth", sample.get("annotation", ""))
    )

    # Convert dict to JSON string if needed
    if isinstance(annotations, dict):
        annotations_text = json.dumps(annotations)
    else:
        annotations_text = str(annotations)

    return convert_to_conversation(image, prompt, annotations_text)


def build_chat_messages(
    instruction: str,
    image: Image.Image = None
) -> List[Dict]:
    """
    Build chat messages for inference (user turn only).

    Args:
        instruction: User instruction/prompt
        image: Optional PIL Image

    Returns:
        List of message dicts for tokenizer.apply_chat_template
    """
    content = []

    if image is not None:
        content.append({"type": "image"})

    content.append({"type": "text", "text": instruction})

    return [{"role": "user", "content": content}]


def format_dict_as_json(data: Dict) -> str:
    """
    Format a dictionary as a pretty-printed JSON string.

    Args:
        data: Dictionary to format

    Returns:
        JSON string with indentation
    """
    return json.dumps(data, indent=2, ensure_ascii=False)


def parse_json_safely(json_str: str) -> Dict:
    """
    Safely parse a JSON string, returning empty dict on failure.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed dict or empty dict if parsing fails
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}
