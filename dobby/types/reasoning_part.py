from typing import Literal, TypedDict


class ReasoningPart(TypedDict):

    type: Literal["reasoning"]

    text: str
