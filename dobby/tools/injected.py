from typing import TYPE_CHECKING, Annotated, get_args, get_origin


class _InjectedMarker:
    """Runtime marker for injected parameters."""

    pass


if TYPE_CHECKING:
    # Type checkers see Injected[T] as just T — enabling autocomplete
    type Injected[T] = T
else:
    # At runtime, Injected[T] produces Annotated[T, _InjectedMarker()]
    # so the detection logic can find and skip it in LLM schemas
    class Injected:
        """Marks a parameter as runtime-injected (hidden from LLM schema).

        Type checkers see the inner type T — enabling autocomplete
        and attribute resolution on the injected context.

        Example:
            async def __call__(
                self,
                ctx: Injected[ToolContext],  # Hidden from LLM, typed as ToolContext
                field: str                   # Visible to LLM
            ) -> dict:
                return {"user_id": ctx.user.id, "field": field}
        """

        def __class_getitem__(cls, item):
            return Annotated[item, _InjectedMarker()]


def is_injected(annotation) -> tuple[bool, type | None]:
    """Check if an annotation is Injected[T].

    Returns:
        Tuple of (is_injected, inner_type).
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        for arg in args[1:]:
            if isinstance(arg, _InjectedMarker):
                return True, args[0] if args else None
    return False, None
