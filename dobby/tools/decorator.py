from collections.abc import Callable

from .schema_utils import process_tool_definition


def tool(
    name: str | None = None, description: str | None = None, version: str = "1.0.0"
) -> Callable[[Callable], Callable]:
    """Decorator to convert a function into a tool with auto-generated schema.

    This creates standalone tools that can be passed to AgentRunner as a list.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        version: Tool version

    Example:
        from typing import Annotated
        from llms.tools import tool, Injected

        @tool(description="Search for documents")
        async def search(
            query: Annotated[str, "Search query string"],
            limit: int = 10
        ) -> list:
            return []

        @tool(description="Get user data")
        async def get_user(
            ctx: Injected[SomeContext],  # Hidden from LLM
            user_id: str  # Visible to LLM
        ) -> dict:
            return {"id": user_id}

        # Usage: Pass to agent
        agent = AgentRunner(provider, tools=[search, get_user])
    """

    def decorator(func: Callable):
        schema, injected_params = process_tool_definition(func, name, description, version)

        # Attach metadata directly to function object
        func._tool_schema = schema
        func._injected_params = injected_params

        return func

    return decorator
