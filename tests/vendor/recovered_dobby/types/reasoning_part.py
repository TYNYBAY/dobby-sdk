from dataclasses import dataclass
from typing import Literal


@dataclass
class ReasoningPart:
    """Reasoning/thinking content from the model."""

    text: str

    kind: Literal["reasoning"] = "reasoning"

    signature: str | None = None
    """The signature of the thinking.

    Supported by:

    * Anthropic (corresponds to the `signature` field)
    * Bedrock (corresponds to the `signature` field)
    * Google (corresponds to the `thought_signature` field)
    * OpenAI (corresponds to the `encrypted_content` field)
    """
