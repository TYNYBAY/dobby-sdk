from typing import Any, Literal, TypedDict


class ToolUsePart(TypedDict):

    type: Literal["tool_use"]

    id: str

    name: str

    inputs: dict[str, Any]
