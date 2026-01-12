"""Base tool abstractions for LLM function calling."""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ParameterType(str, Enum):
    """JSON Schema compatible parameter types."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""

    name: str
    """Name of the parameter."""

    type: ParameterType
    """Data type of the parameter (string, number, integer, boolean, array, object)."""

    description: str
    """Human-readable description of what this parameter is for."""

    required: bool = True
    """Whether this parameter is required or optional."""

    default: Any | None = None
    """Default value if the parameter is not provided."""

    enum: list[Any] | None = None
    """List of allowed values for this parameter."""

    items: dict[str, Any] | None = None
    """Schema for array elements when type is 'array'."""

    properties: dict[str, "ToolParameter"] | None = None
    """Nested parameters for object types."""

    # Constraints
    minimum: int | float | None = None
    """Minimum value for numeric types."""

    maximum: int | float | None = None
    """Maximum value for numeric types."""

    min_length: int | None = None
    """Minimum length for string types."""

    max_length: int | None = None
    """Maximum length for string types."""

    format: str | None = None
    """Format hint for string types (e.g., 'date', 'date-time', 'email', 'uri')."""

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {"type": self.type.value, "description": self.description}

        if self.default is not None:
            schema["default"] = self.default

        if self.enum:
            schema["enum"] = self.enum

        if self.type == ParameterType.ARRAY and self.items:
            schema["items"] = self.items

        if self.type == ParameterType.OBJECT and self.properties:
            schema["properties"] = {
                name: param.to_json_schema() for name, param in self.properties.items()
            }

        # Add constraints
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.format is not None:
            schema["format"] = self.format

        return schema

