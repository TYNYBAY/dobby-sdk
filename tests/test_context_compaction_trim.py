# ruff: noqa: E402
"""Trim tests for the recovered historical editor against current executor shape.

Messages are built the way ``AgentExecutor._emit_tool_result`` writes them:
one assistant message with one ``ToolUsePart``, then one user message with one
``ToolResultPart`` whose text is ``str(result)``. Parallel batches are captured
from the current executor, then rebuilt with the recovered message types so
``edit_context`` can see them. Phase 3 tests are not involved.
"""

# isort: off
from __future__ import annotations

import asyncio
from typing import Annotated, Any
from unittest.mock import AsyncMock

import pytest

from dobby import AgentExecutor
from dobby.tools import Tool
from dobby.types import StreamEndEvent
from dobby.types import TextPart as CurrentTextPart
from dobby.types import ToolResultPart as CurrentToolResultPart
from dobby.types import ToolUsePart as CurrentToolUsePart
from dobby.types import Usage

from tests.compaction_subject import load_recovered

load_recovered()

from recovered_dobby.context import edit_context
from recovered_dobby.context.policy import ContextPolicy
from recovered_dobby.types import AssistantMessagePart
from recovered_dobby.types import TextPart
from recovered_dobby.types import ToolResultPart
from recovered_dobby.types import ToolUsePart
from recovered_dobby.types import UserMessagePart

# isort: on

_LABELS = ("A", "B", "C", "D", "E")
_PLACEHOLDER = "[Tool result cleared to save context.]"


class _EchoTool(Tool):
    name = "echo"
    description = "Return the label it was given."

    def __call__(self, label: Annotated[str, "Label"]) -> dict[str, str]:
        return {"label": label}


class _EmptyDictTool(Tool):
    name = "empty_dict"
    description = "Return an empty object."

    def __call__(self) -> dict[str, str]:
        return {}


class _EmptyStringTool(Tool):
    name = "empty_str"
    description = "Return an empty string."

    def __call__(self) -> str:
        return ""


def _policy(keep_last_n: int) -> ContextPolicy:
    return ContextPolicy(keep_last_n=keep_last_n)


def _use(call_id: str, *, name: str = "echo", inputs: dict[str, Any] | None = None) -> ToolUsePart:
    return ToolUsePart(id=call_id, name=name, inputs={} if inputs is None else inputs)


def _result(call_id: str, text: str, *, name: str = "echo") -> ToolResultPart:
    return ToolResultPart(
        tool_use_id=call_id,
        name=name,
        parts=[TextPart(text=text)],
    )


def _pair(
    call_id: str,
    text: str,
    *,
    name: str = "echo",
    inputs: dict[str, Any] | None = None,
) -> tuple[AssistantMessagePart, UserMessagePart]:
    """One current-executor round-trip: a single use message, then its result."""
    return (
        AssistantMessagePart(parts=[_use(call_id, name=name, inputs=inputs)]),
        UserMessagePart(parts=[_result(call_id, text, name=name)]),
    )


def _labeled_history(
    labels: tuple[str, ...] = _LABELS,
) -> list[AssistantMessagePart | UserMessagePart]:
    messages: list[AssistantMessagePart | UserMessagePart] = []
    for label in labels:
        use, result = _pair(
            f"id-{label}",
            f"result-{label}",
            inputs={"label": label, "query": f"q-{label}"},
        )
        messages.extend((use, result))
    return messages


def _result_parts(messages: list[Any]) -> list[ToolResultPart]:
    return [
        part
        for message in messages
        if isinstance(message, UserMessagePart)
        for part in message.parts
        if isinstance(part, ToolResultPart)
    ]


def _use_parts(messages: list[Any]) -> list[ToolUsePart]:
    return [
        part
        for message in messages
        if isinstance(message, AssistantMessagePart)
        for part in message.parts
        if isinstance(part, ToolUsePart)
    ]


def _result_texts(messages: list[Any]) -> list[str]:
    texts: list[str] = []
    for part in _result_parts(messages):
        texts.append("".join(piece.text for piece in part.parts if isinstance(piece, TextPart)))
    return texts


def _trim(messages: list[Any], keep_last_n: int) -> tuple[list[Any], Any]:
    return edit_context(messages, _policy(keep_last_n))


def _assert_ids_paired(messages: list[Any]) -> None:
    uses = _use_parts(messages)
    results = _result_parts(messages)
    use_ids = [part.id for part in uses]
    for result in results:
        assert result.tool_use_id in use_ids
        match = next(part for part in uses if part.id == result.tool_use_id)
        assert result.name == match.name


def test_basic_trim_tools_a_through_e_keep_last_2() -> None:
    """keep_last_n=2 on tools A-E clears A-C and leaves D and E verbatim."""
    messages = _labeled_history()
    edited, applied = _trim(messages, 2)
    assert applied is not None
    assert applied.type == "clear_tool_uses"
    assert applied.cleared_tool_uses == 3
    assert _result_texts(edited) == [
        _PLACEHOLDER,
        _PLACEHOLDER,
        _PLACEHOLDER,
        "result-D",
        "result-E",
    ]
    _assert_ids_paired(edited)


@pytest.mark.parametrize(
    ("keep_last_n", "expected"),
    [
        pytest.param(0, [_PLACEHOLDER] * 5, id="trim-keep-last-n-0"),
        pytest.param(
            1,
            [_PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, "result-E"],
            id="trim-keep-last-n-1",
        ),
        pytest.param(
            2,
            [_PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, "result-D", "result-E"],
            id="trim-keep-last-n-2",
        ),
        pytest.param(
            5,
            ["result-A", "result-B", "result-C", "result-D", "result-E"],
            id="trim-keep-last-n-equals-n",
        ),
        pytest.param(
            6,
            ["result-A", "result-B", "result-C", "result-D", "result-E"],
            id="trim-keep-last-n-greater-than-n",
        ),
    ],
)
def test_keep_last_n_boundary(keep_last_n: int, expected: list[str]) -> None:
    """keep_last_n retains that many newest completed pairs and clears the rest."""
    messages = _labeled_history()
    edited, applied = _trim(messages, keep_last_n)
    assert _result_texts(edited) == expected
    if keep_last_n >= len(_LABELS):
        assert applied is None
        assert edited is messages
    else:
        assert applied is not None
        assert applied.cleared_tool_uses == len(_LABELS) - keep_last_n


def test_exact_trim_boundary() -> None:
    """The cut sits exactly between pair N-keep-1 and pair N-keep."""
    messages = _labeled_history()
    keep_last_n = 2
    edited, applied = _trim(messages, keep_last_n)
    texts = _result_texts(edited)
    cut = len(_LABELS) - keep_last_n
    assert applied is not None
    assert texts[cut - 1] == _PLACEHOLDER
    assert texts[cut] == "result-D"
    assert texts[:cut] == [_PLACEHOLDER] * cut
    assert texts[cut:] == ["result-D", "result-E"]


def test_tool_use_result_ids_stay_paired() -> None:
    """Clearing a result does not drop or rewrite its tool_use_id."""
    messages = _labeled_history()
    edited, applied = _trim(messages, 2)
    assert applied is not None
    _assert_ids_paired(edited)
    assert [part.id for part in _use_parts(edited)] == [f"id-{label}" for label in _LABELS]
    assert [part.tool_use_id for part in _result_parts(edited)] == [
        f"id-{label}" for label in _LABELS
    ]


def test_tool_call_and_inputs_remain_when_result_cleared() -> None:
    """The tool call message, its inputs, and its metadata stay the same objects."""
    messages = _labeled_history()
    original_uses = _use_parts(messages)
    for part in original_uses:
        part.metadata = {"signature": part.id}
    edited, applied = _trim(messages, 1)
    assert applied is not None
    edited_uses = _use_parts(edited)
    assert edited_uses == original_uses
    for original, edited_use in zip(original_uses, edited_uses, strict=True):
        assert edited_use is original
        assert edited_use.inputs is original.inputs
        assert edited_use.inputs["query"].startswith("q-")
        assert edited_use.metadata == {"signature": original.id}
    assert _result_texts(edited)[:4] == [_PLACEHOLDER] * 4


def test_inflight_tool_use_is_not_a_completed_pair() -> None:
    """A trailing tool call with no result is not counted and is not cleared."""
    messages = _labeled_history()
    inflight = _use("id-F", inputs={"label": "F", "query": "q-F"})
    inflight.metadata = {"open": True}
    messages.append(AssistantMessagePart(parts=[inflight]))
    edited, applied = _trim(messages, 2)
    assert applied is not None
    assert applied.cleared_tool_uses == 3
    assert _result_texts(edited) == [
        _PLACEHOLDER,
        _PLACEHOLDER,
        _PLACEHOLDER,
        "result-D",
        "result-E",
    ]
    assert _use_parts(edited)[-1] is inflight
    assert [part.tool_use_id for part in _result_parts(edited)] == [
        f"id-{label}" for label in _LABELS
    ]
    assert "id-F" not in [part.tool_use_id for part in _result_parts(edited)]


def test_empty_string_result_is_cleared() -> None:
    """An empty-string result is still a completed pair and clears to the placeholder."""
    messages: list[Any] = []
    messages.extend(_pair("id-A", "result-A", inputs={"label": "A"}))
    messages.extend(_pair("id-B", "", inputs={"label": "B"}))
    messages.extend(_pair("id-C", "result-C", inputs={"label": "C"}))
    edited, applied = _trim(messages, 1)
    assert applied is not None
    assert _result_texts(edited) == [_PLACEHOLDER, _PLACEHOLDER, "result-C"]
    assert _result_parts(edited)[1].tool_use_id == "id-B"


def test_empty_string_result_is_kept() -> None:
    """A kept empty-string result stays empty rather than becoming a placeholder."""
    messages: list[Any] = []
    messages.extend(_pair("id-A", "result-A", inputs={"label": "A"}))
    messages.extend(_pair("id-B", "", inputs={"label": "B"}))
    edited, applied = _trim(messages, 1)
    assert applied is not None
    assert _result_texts(edited) == [_PLACEHOLDER, ""]
    assert _result_parts(edited)[1].tool_use_id == "id-B"


def test_empty_structured_result_is_cleared() -> None:
    """``str({})`` is a normal payload and is replaced when that pair is cleared."""
    messages: list[Any] = []
    messages.extend(_pair("id-A", "{}", name="empty_dict", inputs={}))
    messages.extend(_pair("id-B", "result-B", inputs={"label": "B"}))
    edited, applied = _trim(messages, 1)
    assert applied is not None
    assert _result_texts(edited) == [_PLACEHOLDER, "result-B"]
    assert _result_parts(edited)[0].tool_use_id == "id-A"
    assert _result_parts(edited)[0].parts[0].text == _PLACEHOLDER


def test_empty_structured_result_is_kept() -> None:
    """A kept empty object stays ``{}`` and an empty parts list stays empty."""
    messages: list[Any] = []
    messages.extend(_pair("id-A", "result-A", inputs={"label": "A"}))
    messages.extend(_pair("id-B", "{}", name="empty_dict", inputs={}))
    empty_parts = ToolResultPart(tool_use_id="id-C", name="empty_dict", parts=[])
    messages.extend(
        (
            AssistantMessagePart(parts=[_use("id-C", name="empty_dict", inputs={})]),
            UserMessagePart(parts=[empty_parts]),
        )
    )
    edited, applied = _trim(messages, 2)
    assert applied is not None
    assert _result_texts(edited) == [_PLACEHOLDER, "{}", ""]
    kept_structured = _result_parts(edited)[1]
    kept_empty = _result_parts(edited)[2]
    assert kept_structured is _result_parts(messages)[1]
    assert kept_empty is empty_parts
    assert kept_empty.parts == []


def test_multiple_tool_result_parts_share_one_message_pair() -> None:
    """Several results in one user message count as one pair, not one each."""
    uses = [_use(f"id-{label}", inputs={"label": label}) for label in ("A", "B", "C")]
    results = [_result(f"id-{label}", f"result-{label}") for label in ("A", "B", "C")]
    older = [
        AssistantMessagePart(parts=uses),
        UserMessagePart(parts=results),
    ]
    newer_use, newer_result = _pair("id-D", "result-D", inputs={"label": "D"})
    messages = [*older, newer_use, newer_result]
    edited, applied = _trim(messages, 1)
    assert applied is not None
    assert applied.cleared_tool_uses == 3
    assert _result_texts(edited) == [_PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, "result-D"]
    _assert_ids_paired(edited)
    for original in uses:
        assert original in _use_parts(edited)
        assert original.inputs["label"] in {"A", "B", "C"}


def test_keep_last_n_counts_message_pairs_not_individual_results() -> None:
    """On executor-shaped history the count matches tool calls; inside one message it does not."""
    executor_shaped = _labeled_history()
    edited, applied = _trim(executor_shaped, 2)
    assert applied is not None
    assert applied.cleared_tool_uses == 3
    assert _result_texts(edited)[-2:] == ["result-D", "result-E"]

    labels = ("A", "B", "C", "D", "E")
    packed = [
        AssistantMessagePart(
            parts=[_use(f"id-{label}", inputs={"label": label}) for label in labels]
        ),
        UserMessagePart(parts=[_result(f"id-{label}", f"result-{label}") for label in labels]),
    ]
    packed_edited, packed_applied = _trim(packed, 2)
    assert packed_applied is None
    assert packed_edited is packed
    assert _result_texts(packed_edited) == [f"result-{label}" for label in labels]
    cleared, cleared_applied = _trim(packed, 0)
    assert cleared_applied is not None
    assert cleared_applied.cleared_tool_uses == 5
    assert _result_texts(cleared) == [_PLACEHOLDER] * 5


def test_caller_owned_messages_are_not_mutated() -> None:
    """The caller's list, kept objects, and original result text stay unchanged."""
    messages = _labeled_history()
    snapshot_ids = [id(message) for message in messages]
    original_texts = [part.parts[0].text for part in _result_parts(messages)]
    original_inputs = [part.inputs for part in _use_parts(messages)]
    edited, applied = _trim(messages, 2)
    assert applied is not None
    assert edited is not messages
    assert [id(message) for message in messages] == snapshot_ids
    assert [part.parts[0].text for part in _result_parts(messages)] == original_texts
    assert [part.inputs for part in _use_parts(messages)] == original_inputs
    assert _result_texts(messages) == [f"result-{label}" for label in _LABELS]
    for index, message in enumerate(messages):
        if index >= (len(_LABELS) - 2) * 2:
            assert edited[index] is message


def _scripted_provider(tool_calls: list[CurrentToolUsePart]) -> Any:
    call_count = 0

    async def mock_chat(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        parts = tool_calls if call_count == 1 else []

        async def stream() -> Any:
            yield StreamEndEvent(
                type="stream_end",
                model="mock",
                parts=parts,
                stop_reason="tool_use" if parts else "end_turn",
                usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        return stream()

    provider = AsyncMock()
    provider.chat = mock_chat
    provider.name = "scripted"
    return provider


def _capture_executor_messages(
    tool_calls: list[CurrentToolUsePart],
    tools: list[Tool],
) -> list[Any]:
    executor = AgentExecutor("openai", _scripted_provider(tool_calls), tools=tools)
    holder: dict[str, list[Any]] = {}
    original = executor._emit_tool_result

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        holder["messages"] = args[3]
        return original(*args, **kwargs)

    executor._emit_tool_result = wrapped  # type: ignore[method-assign]

    async def run() -> None:
        async for _event in executor.run_stream([]):
            pass

    asyncio.run(run())
    return holder["messages"]


def _project_executor_messages(
    messages: list[Any],
) -> list[AssistantMessagePart | UserMessagePart]:
    """Rebuild current executor messages with the recovered dataclasses."""
    projected: list[AssistantMessagePart | UserMessagePart] = []
    for message in messages:
        if message.role == "assistant":
            parts: list[Any] = []
            for part in message.parts:
                if isinstance(part, CurrentToolUsePart):
                    parts.append(
                        ToolUsePart(
                            id=part.id,
                            name=part.name,
                            inputs=dict(part.inputs),
                            metadata=None if part.metadata is None else dict(part.metadata),
                        )
                    )
                elif isinstance(part, CurrentTextPart):
                    parts.append(TextPart(text=part.text))
            projected.append(AssistantMessagePart(parts=parts))
        else:
            parts = []
            for part in message.parts:
                if isinstance(part, CurrentToolResultPart):
                    inner = [
                        TextPart(text=piece.text)
                        for piece in part.parts
                        if isinstance(piece, CurrentTextPart)
                    ]
                    parts.append(
                        ToolResultPart(
                            tool_use_id=part.tool_use_id,
                            name=part.name,
                            parts=inner,
                            is_error=part.is_error,
                        )
                    )
                elif isinstance(part, CurrentTextPart):
                    parts.append(TextPart(text=part.text))
            projected.append(UserMessagePart(parts=parts))
    return projected


@pytest.mark.parametrize(
    "batch_size", [2, 3, 5, 10, 50], ids=lambda size: f"trim-parallel-batch-{size}"
)
def test_parallel_batch_keep_last_2(batch_size: int) -> None:
    """Current parallel batches are one message pair per tool call, trimmed that way."""
    calls = [
        CurrentToolUsePart(id=f"id-{index}", name="echo", inputs={"label": str(index)})
        for index in range(batch_size)
    ]
    live = _capture_executor_messages(calls, [_EchoTool()])
    assert len(live) == batch_size * 2
    for index in range(batch_size):
        use_message = live[index * 2]
        result_message = live[index * 2 + 1]
        assert isinstance(use_message, type(live[0]))
        assert use_message.role == "assistant"
        assert len(use_message.parts) == 1
        assert isinstance(use_message.parts[0], CurrentToolUsePart)
        assert use_message.parts[0].id == f"id-{index}"
        assert result_message.role == "user"
        assert len(result_message.parts) == 1
        result = result_message.parts[0]
        assert isinstance(result, CurrentToolResultPart)
        assert result.tool_use_id == f"id-{index}"
        assert result.parts[0].text == str({"label": str(index)})

    edited, applied = _trim(_project_executor_messages(live), 2)
    texts = _result_texts(edited)
    assert len(texts) == batch_size
    if batch_size <= 2:
        assert applied is None
        assert texts == [str({"label": str(index)}) for index in range(batch_size)]
    else:
        assert applied is not None
        assert applied.cleared_tool_uses == batch_size - 2
        assert texts[:-2] == [_PLACEHOLDER] * (batch_size - 2)
        assert texts[-2:] == [
            str({"label": str(batch_size - 2)}),
            str({"label": str(batch_size - 1)}),
        ]
    _assert_ids_paired(edited)


def test_executor_empty_string_and_empty_object_survive_as_pairs() -> None:
    """Current ``str(result)`` text is what trim sees for empty string and ``{}``."""
    empty_calls = [
        CurrentToolUsePart(id="id-empty-str", name="empty_str", inputs={}),
        CurrentToolUsePart(id="id-empty-dict", name="empty_dict", inputs={}),
        CurrentToolUsePart(id="id-keep", name="echo", inputs={"label": "keep"}),
    ]
    live = _capture_executor_messages(
        empty_calls,
        [_EmptyStringTool(), _EmptyDictTool(), _EchoTool()],
    )
    projected = _project_executor_messages(live)
    assert _result_texts(projected) == ["", "{}", str({"label": "keep"})]
    edited, applied = _trim(projected, 1)
    assert applied is not None
    assert _result_texts(edited) == [_PLACEHOLDER, _PLACEHOLDER, str({"label": "keep"})]
    assert [part.tool_use_id for part in _result_parts(edited)] == [
        "id-empty-str",
        "id-empty-dict",
        "id-keep",
    ]
