class Injected[T]:
    """Marks a parameter as runtime-injected (hidden from LLM schema).

    This is a type marker that tells the tool decorator that a parameter
    should not be included in the schema sent to the LLM, but will instead
    be provided at runtime by the agent runner.

    Example:
        @agent.tool(description="Get user info")
        async def get_user(
            ctx: Injected[ToolContext],  # Hidden from LLM
            field: str                   # Visible to LLM
        ) -> dict:
            return {"user_id": ctx.user.id, "field": field}
    """

    pass
