from typing import Literal, TypedDict


class TextPart(TypedDict):

    type: Literal["text"]

    text: str
