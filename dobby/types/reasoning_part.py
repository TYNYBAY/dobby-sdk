from dataclasses import dataclass
from typing import Literal


@dataclass
class ReasoningPart:
    """Reasoning/thinking content from the model."""

    text: str

    kind: Literal["reasoning"] = "reasoning"
