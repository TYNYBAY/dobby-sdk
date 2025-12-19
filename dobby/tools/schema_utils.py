"""Schema generation utilities for tool decorators and runners.

This module consolidates all schema generation logic to eliminate code duplication
between decorator.py and runner.py.
"""

from collections.abc import Callable
from datetime import date, datetime
import inspect
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .base import ParameterType, ToolParameter, ToolSchema
from .injected import Injected


def is_typeddict(tp: type) -> bool:
    # TypedDict is a subclass of dict and has __annotations__
    return isinstance(tp, type) and issubclass(tp, dict) and hasattr(tp, "__annotations__")


def is_injected_parameter(annotation) -> tuple[bool, type | None]:
    """Check if parameter is wrapped with Injected[T].

    Args:
        annotation: Type annotation to check

    Returns:
        Tuple of (is_injected, inner_type)
    """
    origin = get_origin(annotation)
    if origin is Injected:
        args = get_args(annotation)
        return True, args[0] if args else None
    return False, None


def extract_pydantic_properties(model_class: type[BaseModel]) -> dict[str, ToolParameter]:
    """Extract properties from a Pydantic model using its JSON schema."""
    properties = {}
    json_schema = model_class.model_json_schema()

    for field_name, field_schema in json_schema.get("properties", {}).items():
        if "$ref" in field_schema:
            ref_path = field_schema["$ref"]
            if ref_path.startswith("#/$defs/"):
                ref_name = ref_path.split("/")[-1]
                if ref_name in json_schema.get("$defs", {}):
                    field_schema = json_schema["$defs"][ref_name]

        param = json_schema_to_tool_parameter(
            field_name, field_schema, json_schema.get("$defs", {})
        )
        properties[field_name] = param

    return properties


def json_schema_to_tool_parameter(
    field_name: str, field_schema: dict, defs: dict = None
) -> ToolParameter:
    """Convert a JSON schema field to ToolParameter."""
    actual_schema = field_schema
    is_optional = False

    if "anyOf" in field_schema:
        for schema_variant in field_schema["anyOf"]:
            if schema_variant.get("type") != "null":
                actual_schema = schema_variant
            else:
                is_optional = True

    if "enum" in field_schema and "anyOf" not in field_schema:
        actual_schema = field_schema

    json_type = actual_schema.get("type", "object")
    param_type = json_schema_type_to_parameter_type(json_type)

    description = actual_schema.get(
        "description", field_schema.get("description", f"Field {field_name}")
    )
    required = not is_optional and "default" not in field_schema

    properties = None
    if param_type == ParameterType.OBJECT and "properties" in actual_schema:
        properties = {}
        for prop_name, prop_schema in actual_schema["properties"].items():
            properties[prop_name] = json_schema_to_tool_parameter(prop_name, prop_schema, defs)

    items = None
    if param_type == ParameterType.ARRAY and "items" in actual_schema:
        items_schema = actual_schema["items"]
        if "$ref" in items_schema:
            ref_path = items_schema["$ref"]
            if ref_path.startswith("#/$defs/") and defs:
                ref_name = ref_path.split("/")[-1]
                if ref_name in defs:
                    items_schema = defs[ref_name]

        if "type" in items_schema:
            items = items_schema
        else:
            items = {"type": "object", "properties": {}}

    param = ToolParameter(
        name=field_name,
        type=param_type,
        description=description,
        required=required,
        properties=properties,
        items=items,
        default=field_schema.get("default"),
    )

    if "minimum" in actual_schema:
        param.minimum = actual_schema["minimum"]
    if "exclusiveMinimum" in actual_schema:
        param.minimum = actual_schema["exclusiveMinimum"]
    if "maximum" in actual_schema:
        param.maximum = actual_schema["maximum"]
    if "exclusiveMaximum" in actual_schema:
        param.maximum = actual_schema["exclusiveMaximum"]
    if "minLength" in actual_schema:
        param.min_length = actual_schema["minLength"]
    if "maxLength" in actual_schema:
        param.max_length = actual_schema["maxLength"]
    if "enum" in actual_schema:
        param.enum = actual_schema["enum"]
    if "format" in actual_schema:
        param.format = actual_schema["format"]

    return param


def json_schema_type_to_parameter_type(json_type: str) -> ParameterType:
    """Convert JSON schema type to ParameterType."""
    type_mapping = {
        "string": ParameterType.STRING,
        "number": ParameterType.NUMBER,
        "integer": ParameterType.INTEGER,
        "boolean": ParameterType.BOOLEAN,
        "array": ParameterType.ARRAY,
        "object": ParameterType.OBJECT,
    }
    return type_mapping.get(json_type, ParameterType.OBJECT)


def extract_parameter_from_type(
    param_name: str, annotation: type, default: Any, description: str
) -> ToolParameter:
    """Extract ToolParameter from a type annotation."""
    is_optional = False
    origin = get_origin(annotation)
    actual_type = annotation

    if origin is not None:
        args = get_args(annotation)
        # Handle Literal types (e.g., Literal["text", "code", "sheet"])
        if origin is Literal:
            has_default = default is not inspect.Parameter.empty
            return ToolParameter(
                name=param_name,
                type=ParameterType.STRING,
                description=description,
                required=not has_default,
                enum=list(args),
                default=default if has_default else None,
            )
        elif len(args) == 2 and type(None) in args:
            is_optional = True
            actual_type = args[0] if args[1] is type(None) else args[1]
            # Recalculate origin after unwrapping!
            origin = get_origin(actual_type)
            # Check if unwrapped type is Literal
            if origin is Literal:
                has_default = default is not inspect.Parameter.empty
                return ToolParameter(
                    name=param_name,
                    type=ParameterType.STRING,
                    description=description,
                    required=False,
                    enum=list(get_args(actual_type)),
                    default=default if has_default else None,
                )
        elif len(args) > 2:
            raise NotImplementedError

    has_default = default is not inspect.Parameter.empty
    required = not has_default and not is_optional

    if inspect.isclass(actual_type) and issubclass(actual_type, BaseModel):
        return ToolParameter(
            name=param_name,
            type=ParameterType.OBJECT,
            description=description,
            required=required,
            properties=extract_pydantic_properties(actual_type),
            default=default if has_default else None,
        )

    if origin is list:
        args = get_args(actual_type)
        if args:
            item_type = args[0]

            if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
                item_properties = extract_pydantic_properties(item_type)
                return ToolParameter(
                    name=param_name,
                    type=ParameterType.ARRAY,
                    description=description,
                    required=required,
                    items={
                        "type": "object",
                        "properties": {
                            name: param.to_json_schema() for name, param in item_properties.items()
                        },
                        "required": [
                            name for name, param in item_properties.items() if param.required
                        ],
                    },
                    default=default if has_default else None,
                )
            elif is_typeddict(item_type):
                props = {}
                for name, ann in item_type.__annotations__.items():
                    # reuse extract_parameter_from_type to produce a ToolParameter for each field
                    tp = extract_parameter_from_type(
                        name, ann, inspect.Parameter.empty, f"Field {name}"
                    )
                    props[name] = tp.to_json_schema()
                required_keys = list(props.keys())
                return ToolParameter(
                    name=param_name,
                    type=ParameterType.ARRAY,
                    description=description,
                    required=required,
                    items={
                        "type": "object",
                        "properties": props,
                        "required": required_keys,
                    },
                    default=default if has_default else None,
                )

            elif item_type is dict or get_origin(item_type) is dict:
                # For list[dict]
                # This should ideally have properties defined via description/metadata
                # This is the issue - list[dict] has no way to define structure!
                return ToolParameter(
                    name=param_name,
                    type=ParameterType.ARRAY,
                    description=description,
                    required=required,
                    items={"type": "object"},
                    default=default if has_default else None,
                )
            else:
                item_param_type = python_type_to_parameter_type(item_type)
                return ToolParameter(
                    name=param_name,
                    type=ParameterType.ARRAY,
                    description=description,
                    required=required,
                    items={"type": item_param_type.value},
                    default=default if has_default else None,
                )

    return ToolParameter(
        name=param_name,
        type=python_type_to_parameter_type(actual_type),
        description=description,
        required=required,
        default=default if has_default else None,
    )


def python_type_to_parameter_type(py_type: type) -> ParameterType:
    """Convert Python type to ParameterType."""
    origin = get_origin(py_type)
    if origin is Union:
        args = get_args(py_type)
        if len(args) == 2 and type(None) in args:
            py_type = args[0] if args[1] is type(None) else args[1]

    if py_type is str:
        return ParameterType.STRING
    elif py_type is int:
        return ParameterType.INTEGER
    elif py_type is float:
        return ParameterType.NUMBER
    elif py_type is bool:
        return ParameterType.BOOLEAN
    elif py_type is list or get_origin(py_type) is list:
        return ParameterType.ARRAY
    elif py_type is dict or get_origin(py_type) is dict:
        return ParameterType.OBJECT
    elif py_type is date or py_type is datetime:
        return ParameterType.STRING
    else:
        return ParameterType.OBJECT


def extract_parameter_from_annotation(
    param_name: str, annotation: type, default: Any
) -> ToolParameter:
    """Extract ToolParameter from function parameter annotation."""
    description = f"Parameter {param_name}"
    actual_type = annotation
    constraints = {}

    if hasattr(annotation, "__metadata__"):
        actual_type = annotation.__origin__

        for metadata in annotation.__metadata__:
            if isinstance(metadata, str):
                description = metadata
            elif isinstance(metadata, FieldInfo):
                if metadata.description:
                    description = metadata.description
                if hasattr(metadata, "ge") and metadata.ge is not None:
                    constraints["minimum"] = metadata.ge
                if hasattr(metadata, "gt") and metadata.gt is not None:
                    constraints["minimum"] = metadata.gt
                if hasattr(metadata, "le") and metadata.le is not None:
                    constraints["maximum"] = metadata.le
                if hasattr(metadata, "lt") and metadata.lt is not None:
                    constraints["maximum"] = metadata.lt
                if hasattr(metadata, "min_length") and metadata.min_length is not None:
                    constraints["min_length"] = metadata.min_length
                if hasattr(metadata, "max_length") and metadata.max_length is not None:
                    constraints["max_length"] = metadata.max_length
                if hasattr(metadata, "enum") and metadata.enum is not None:
                    constraints["enum"] = metadata.enum
                if metadata.default is not None:
                    default = metadata.default

    param = extract_parameter_from_type(param_name, actual_type, default, description)

    for key, value in constraints.items():
        setattr(param, key, value)

    return param


def process_tool_definition(
    func: Callable, name: str | None, description: str | None, version: str = "1.0.0"
) -> tuple[ToolSchema, dict]:
    """Shared tool processing logic for both standalone and agent-bound decorators.

    This function extracts all the common logic for processing a function into a tool,
    including parameter extraction, schema generation, and injected parameter handling.

    Args:
        func: Function to process into a tool
        name: Tool name (optional, defaults to function name)
        description: Tool description (optional, uses docstring if not provided)
        version: Tool version

    Returns:
        tuple: (ToolSchema for LLM, injected_params dict for runtime)
    """
    func_name = name or func.__name__
    sig = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    tool_description = (
        description or (func.__doc__ and func.__doc__.strip()) or f"Function {func_name}"
    )

    visible_params = []
    injected_params = {}

    for param_name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        if param_name in type_hints:
            annotation = type_hints[param_name]
        else:
            annotation = str

        is_injected_param, inner_type = is_injected_parameter(annotation)

        if is_injected_param:
            injected_params[param_name] = inner_type
        else:
            tool_param = extract_parameter_from_annotation(
                param_name=param_name, annotation=annotation, default=param.default
            )
            if tool_param:
                visible_params.append(tool_param)

    # Create schema with only visible parameters
    schema = ToolSchema(
        name=func_name,
        description=tool_description,
        parameters=visible_params,  # Only non-injected params
        version=version,
    )

    return schema, injected_params
