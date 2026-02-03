"""Tool class for defining LLM-callable tools.

This module provides the Tool base class that all tools should inherit from.
Tools are defined as dataclasses with a __call__ method that contains the
tool's implementation.

Example:
    from dataclasses import dataclass
    from typing import Annotated
    from dobby.tools import Tool, Injected

    @dataclass
    class FetchPolicyTool(Tool):
        name = "fetch_policy"
        description = "Fetch policy details by ID"

        async def __call__(
            self,
            ctx: Injected[RunToolContext],
            policy_id: Annotated[str, "The policy ID to fetch"],
        ) -> dict:
            return await ctx.db.get_policy(policy_id)
"""

from dataclasses import dataclass
import inspect
from typing import Any, ClassVar, get_type_hints

from google.genai import types as genai_types
from openai.types.responses import FunctionToolParam
from pydantic import BaseModel

from .base import ToolParameter
from .injected import is_injected
from .schema_utils import process_tool_definition


@dataclass
class Tool:
    """Base class for all tools. Subclass and implement __call__.

    The schema is auto-generated from the __call__ method's type annotations.
    Use Annotated[type, "description"] for parameter descriptions.
    Use Injected[T] for the first parameter to inject runtime context.

    Class Attributes (define in subclass as class variables, NOT fields):
        name: Tool name (defaults to class name if not set)
        description: Tool description for the LLM
        max_retries: Maximum retry attempts on failure (default: 1)
        requires_approval: Whether tool needs human approval before execution (default: False)
        stream_output: Whether tool yields streaming events (default: False)
        terminal: Whether tool exits the agent loop (default: False)

    Terminal Tools:
        When terminal=True, the AgentExecutor will:
        1. Execute the tool
        2. Yield a ToolResultEvent with is_terminal=True
        3. Exit the loop without sending result to LLM

        Use for actions that end the conversation:
        - end_call, hang_up
        - transfer_to_human
        - escalate_to_supervisor
    """

    # Class Variables: These define tool configuration on the class itself.
    # Using ClassVar ensures @dataclass doesn't treat them as instance fields (args to __init__).
    name: ClassVar[str] = ""  # Falls back to class name if None
    description: ClassVar[str]  # Required! No default
    max_retries: ClassVar[int] = 1
    requires_approval: ClassVar[bool] = False
    stream_output: ClassVar[bool] = False
    terminal: ClassVar[bool] = False
    """If True, executing this tool exits the agent loop and returns control to caller."""

    # Auto-generated class variables (set by __init_subclass__)
    _parameters: ClassVar[list[ToolParameter]]
    takes_ctx: ClassVar[bool] = False

    # Instance attributes
    _model: type[BaseModel] | None = None

    def __init_subclass__(cls, **kwargs):
        """Auto-generate schema when a Tool subclass is defined."""
        super().__init_subclass__(**kwargs)

        # Only process if __call__ is overridden (not the base class implementation)
        if "__call__" in cls.__dict__:
            if not getattr(cls, "description", None):
                raise TypeError(
                    f"Tool '{cls.__name__}' must define a 'description' class attribute."
                )

            # Validate streaming tools are async generators
            if getattr(cls, "stream_output", False):
                if not inspect.isasyncgenfunction(cls.__call__):
                    raise TypeError(
                        f"Tool '{cls.__name__}' has stream_output=True but __call__ "
                        "is not an async generator. Use 'async def' with 'yield'."
                    )

            parameters, takes_ctx = cls._generate_schema()
            cls._parameters = parameters
            cls.takes_ctx = takes_ctx

    @classmethod
    def _generate_schema(cls) -> tuple[list[ToolParameter], bool]:
        """Generate parameters from __call__ signature.

        Returns:
            Tuple of (list[ToolParameter], takes_ctx) where takes_ctx indicates
            if the first parameter is Injected[T].
        """
        # Check if first param (after self) is Injected[T]
        takes_ctx = False
        try:
            hints = get_type_hints(cls.__call__, include_extras=True)
            # Remove 'return' and 'self' from hints
            param_names = [k for k in hints.keys() if k not in ("return",)]

            if param_names:
                first_param_type = hints[param_names[0]]
                is_inj, _ = is_injected(first_param_type)
                if is_inj:
                    takes_ctx = True
        except Exception:
            # If type hints fail, assume no context
            pass

        # Use class attribute for name, fall back to class name
        tool_name = getattr(cls, "name", None) or cls.__name__
        cls.name = tool_name  # Ensure name is always set on the class

        tool_description = getattr(cls, "description", None)

        parameters, _ = process_tool_definition(
            cls.__call__, description=tool_description, version="1.0.0"
        )

        return parameters, takes_ctx

    @classmethod
    def from_model(
        cls, model: type[BaseModel], name: str, description: str | None = None
    ) -> "Tool":
        """Create a Tool instance from a Pydantic model using its raw schema.

        Args:
            model: Pydantic model defining the schema
            name: Tool name
            description: Tool description (defaults to model description)

        Returns:
            Configured Tool instance
        """
        # Capture description in local var to avoid class scope confusion
        tool_description = description or model.model_json_schema().get("description", "") or ""

        # dynamic subclass to satisfy proper instantiation
        class ModelTool(Tool):
            description = tool_description

            def __call__(self, **kwargs):
                pass

        ModelTool.name = name
        tool = ModelTool()
        tool._model = model
        return tool

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool. Override in subclass.

        Args:
            *args: Positional arguments (first may be context if takes_ctx=True)
            **kwargs: Tool parameters from LLM

        Returns:
            Tool result to send back to LLM
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__ method")

    def to_openai_format(self) -> FunctionToolParam:
        """Get tool definition in OpenAI format."""
        # TODO: this could be simplifed frst is we simplify in schema_utils.py
        if self._model:
            parameters = self._model.model_json_schema()
        else:
            parameters = {
                "type": "object",
                "properties": {param.name: param.to_json_schema() for param in self._parameters},
                "required": [p.name for p in self._parameters if p.required],
            }

        # TODO: improve type handling inthis code
        return FunctionToolParam(
            type="function",
            name=self.name,
            description=self.description,
            parameters=parameters,
        )

    def to_anthropic_format(self) -> dict[str, Any]:
        """Get tool definition in Anthropic format."""
        if self._model:
            input_schema = self._model.model_json_schema()
        else:
            input_schema = {
                "type": "object",
                "properties": {param.name: param.to_json_schema() for param in self._parameters},
                "required": [p.name for p in self._parameters if p.required],
            }

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema,
        }

    def to_gemini_format(self) -> genai_types.Tool:
        """Get tool definition in Gemini Tool format.

        Returns a genai_types.Tool containing this tool's FunctionDeclaration.
        Can be passed directly to GeminiProvider.chat(tools=[...]).

        Example:
            tools = [my_tool.to_gemini_format()]
            await provider.chat(messages, tools=tools)
        """
        if self._model:
            parameters = self._model.model_json_schema()
        else:
            # TODO: improve type handling inthis code
            parameters = {
                "type": "object",
                "properties": {param.name: param.to_json_schema() for param in self._parameters},
                "required": [p.name for p in self._parameters if p.required],
            }

        func_decl = genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=parameters,
        )
        return genai_types.Tool(function_declarations=[func_decl])
