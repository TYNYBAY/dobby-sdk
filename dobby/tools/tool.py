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

from dataclasses import dataclass, field
import inspect
from typing import Any, get_origin, get_type_hints

from .base import ToolSchema
from .injected import Injected
from .schema_utils import process_tool_definition


@dataclass
class Tool:
    """Base class for all tools. Subclass and implement __call__.
    
    The schema is auto-generated from the __call__ method's type annotations.
    Use Annotated[type, "description"] for parameter descriptions.
    Use Injected[T] for the first parameter to inject runtime context.
    
    Attributes:
        name: Tool name (defaults to class name)
        description: Tool description for the LLM
        max_retries: Maximum retry attempts on failure
        requires_approval: Whether tool needs human approval before execution
        stream_output: Whether tool yields streaming events (requires async generator)
    """
    
    # Auto-generated from __call__ signature
    _tool_schema: ToolSchema = field(init=False, repr=False)
    takes_ctx: bool = field(init=False, default=False)
    
    # Overridable class attributes
    name: str | None = None
    description: str | None = None
    max_retries: int = 1
    requires_approval: bool = False
    stream_output: bool = False
    
    def __init_subclass__(cls, **kwargs):
        """Auto-generate schema when a Tool subclass is defined."""
        super().__init_subclass__(**kwargs)
        
        # Only process if __call__ is defined and not the base class
        if hasattr(cls, '__call__') and cls.__call__ is not Tool.__call__:
            # Validate streaming tools are async generators
            if cls.stream_output:
                if not inspect.isasyncgenfunction(cls.__call__):
                    raise TypeError(
                        f"Tool '{cls.__name__}' has stream_output=True but __call__ "
                        "is not an async generator. Use 'async def' with 'yield'."
                    )
            
            schema, takes_ctx = cls._generate_schema()
            cls._tool_schema = schema
            cls.takes_ctx = takes_ctx
    
    @classmethod
    def _generate_schema(cls) -> tuple[ToolSchema, bool]:
        """Generate ToolSchema from __call__ signature.
        
        Returns:
            Tuple of (ToolSchema, takes_ctx) where takes_ctx indicates
            if the first parameter is Injected[T].
        """
        # Check if first param (after self) is Injected[T]
        takes_ctx = False
        try:
            hints = get_type_hints(cls.__call__)
            # Remove 'return' and 'self' from hints
            param_names = [k for k in hints.keys() if k not in ('return',)]
            
            if param_names:
                first_param_type = hints[param_names[0]]
                origin = get_origin(first_param_type)
                if origin is Injected:
                    takes_ctx = True
        except Exception:
            # If type hints fail, assume no context
            pass
        
        schema, _ = process_tool_definition(
            cls.__call__,
            name=getattr(cls, 'name', None) or cls.__name__,
            description=getattr(cls, 'description', None),
            version="1.0.0"
        )
        
        return schema, takes_ctx
    
    async def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool. Override in subclass.
        
        Args:
            *args: Positional arguments (first may be context if takes_ctx=True)
            **kwargs: Tool parameters from LLM
            
        Returns:
            Tool result to send back to LLM
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__ method"
        )
    
    def to_openai_format(self):
        """Get tool definition in OpenAI format."""
        return self._tool_schema.to_openai_format()
    
    def to_anthropic_format(self):
        """Get tool definition in Anthropic format."""
        return self._tool_schema.to_anthropic_format()
