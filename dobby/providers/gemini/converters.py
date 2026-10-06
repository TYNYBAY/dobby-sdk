"""Converters for Gemini message and tool formats.

This module handles bidirectional conversion between Dobby's provider-agnostic
types and Google Gemini's native types.
"""

import base64
from collections.abc import Iterable
from typing import Any

from google.genai import types as genai_types

from ...types import (
    AssistantMessagePart,
    Base64ImageSource,
    Base64PDFSource,
    DocumentPart,
    FileDocumentSource,
    ImagePart,
    MessagePart,
    PlainTextSource,
    ReasoningPart,
    TextPart,
    ToolResultPart,
    ToolUsePart,
    URLImageSource,
    URLSource,
    UserMessagePart,
)


def _text_to_gemini(part: TextPart) -> genai_types.Part:
    """Convert TextPart to Gemini Part."""
    return genai_types.Part.from_text(text=part.text)


def _image_to_gemini(part: ImagePart) -> genai_types.Part:
    """Convert ImagePart to Gemini Part.

    Handles both URL-based and base64-encoded images.
    """
    match part.source:
        case URLImageSource(url=url):
            return genai_types.Part.from_uri(file_uri=url, mime_type="image/jpeg")
        case Base64ImageSource(data=data, media_type=media_type):
            # Decode base64 to bytes for Gemini
            image_bytes = base64.b64decode(data)
            return genai_types.Part.from_bytes(data=image_bytes, mime_type=media_type)
    raise ValueError(f"Unknown image source type: {part.source}")


def _document_to_gemini(part: DocumentPart) -> genai_types.Part:
    """Convert DocumentPart to Gemini Part.

    Handles URL, base64 PDF, plain text, and file ID sources.
    """
    match part.source:
        case URLSource(url=url):
            return genai_types.Part.from_uri(file_uri=url, mime_type="application/pdf")
        case Base64PDFSource(data=data, media_type=media_type):
            pdf_bytes = base64.b64decode(data)
            return genai_types.Part.from_bytes(data=pdf_bytes, mime_type=media_type)
        case PlainTextSource(data=data):
            # Plain text as inline text part
            return genai_types.Part.from_text(text=f"[Document: {part.filename}]\n{data}")
        case FileDocumentSource(file_id=file_id):
            # Gemini uses file URIs for uploaded files
            return genai_types.Part.from_uri(
                file_uri=f"https://generativelanguage.googleapis.com/v1beta/files/{file_id}",
                mime_type="application/pdf",
            )
    raise ValueError(f"Unknown document source type: {part.source}")


def _tool_use_to_gemini(part: ToolUsePart) -> genai_types.Part:
    """Convert ToolUsePart to Gemini Part with FunctionCall.

    Creates a Part containing a FunctionCall for the model's tool invocation.
    Also restores thought and thought_signature if present in metadata.
    """
    part_obj = genai_types.Part(
        function_call=genai_types.FunctionCall(
            name=part.name,
            args=part.inputs,
        )
    )

    if part.metadata and (signature := part.metadata.get("signature")):
        part_obj.thought_signature = base64.b64decode(signature)

    return part_obj


def _tool_result_to_gemini(part: ToolResultPart) -> genai_types.Part:
    """Convert ToolResultPart to Gemini Part with FunctionResponse.

    Creates a Part containing a FunctionResponse with the tool's output.
    """
    # Combine all result parts into a single response value
    result_data: dict[str, Any] = {}

    for p in part.parts:
        match p:
            case TextPart(text=text):
                result_data["text"] = text
            case _:
                # For other types, convert to string representation
                result_data["content"] = str(p)

    if part.is_error:
        result_data["error"] = True

    return genai_types.Part(
        function_response=genai_types.FunctionResponse(
            name=part.name,
            response=result_data,
        )
    )


def content_part_to_gemini(
    part: TextPart | ImagePart | DocumentPart,
) -> genai_types.Part:
    """Convert a content part to Gemini format.

    Args:
        part: A Dobby content part (TextPart, ImagePart, or DocumentPart)

    Returns:
        Gemini Part object
    """
    match part:
        case TextPart():
            return _text_to_gemini(part)
        case ImagePart():
            return _image_to_gemini(part)
        case DocumentPart():
            return _document_to_gemini(part)
    raise ValueError(f"Unknown content part type: {part}")


def _has_function_response(content: genai_types.Content) -> bool:
    """True when ``content`` carries a tool result.

    Function responses must stay on their own user turn, immediately after the
    model function call. Plain user text can be merged without disturbing that.
    """
    return any(
        getattr(part, "function_response", None) is not None for part in content.parts or []
    )


def _merge_consecutive_same_role_contents(
    contents: list[genai_types.Content],
) -> list[genai_types.Content]:
    """Collapse adjacent same-role turns so Gemini sees alternating roles.

    Summarization can insert a ``<summary>`` user message next to a leading
    user prompt, and can leave assistant text immediately before a later
    tool-use. Gemini rejects consecutive same-role contents. Tool-result turns
    are left separate so a function response still follows its function call.
    """
    merged: list[genai_types.Content] = []
    for content in contents:
        if (
            merged
            and merged[-1].role == content.role
            and not _has_function_response(merged[-1])
            and not _has_function_response(content)
        ):
            merged[-1] = genai_types.Content(
                role=content.role,
                parts=[*(merged[-1].parts or []), *(content.parts or [])],
            )
            continue
        merged.append(content)
    return merged


def to_gemini_messages(
    messages: Iterable[MessagePart],
) -> list[genai_types.Content]:
    """Convert provider-agnostic messages to Gemini Content format.

    Handles conversion of different message types:
    - UserMessagePart → Content(role='user', parts=[...])
    - AssistantMessagePart → Content(role='model', parts=[...])
    - ToolUsePart → Part with FunctionCall
    - ToolResultPart → Part with FunctionResponse

    Adjacent same-role turns are merged so a summary inserted after the
    opening user message, or assistant text left before a kept tool-use, does
    not produce consecutive roles. Function responses stay on their own user
    turn.

    Args:
        messages: Iterable of Dobby MessagePart objects

    Returns:
        List of Gemini Content objects ready for API call
    """
    gemini_contents: list[genai_types.Content] = []

    for message in messages:
        match message:
            case UserMessagePart(parts=parts):
                gemini_parts: list[genai_types.Part] = []
                function_responses: list[genai_types.Part] = []

                for p in parts:
                    if isinstance(p, ToolResultPart):
                        # Function responses go in separate content
                        function_responses.append(_tool_result_to_gemini(p))
                    else:
                        gemini_parts.append(content_part_to_gemini(p))

                # Add user content if we have regular parts
                if gemini_parts:
                    gemini_contents.append(genai_types.Content(role="user", parts=gemini_parts))

                # Add function responses as user content (per Gemini spec)
                if function_responses:
                    gemini_contents.append(
                        genai_types.Content(role="user", parts=function_responses)
                    )

            case AssistantMessagePart(parts=parts):
                gemini_parts: list[genai_types.Part] = []

                for p in parts:
                    match p:
                        case TextPart():
                            gemini_parts.append(_text_to_gemini(p))
                        case ToolUsePart():
                            gemini_parts.append(_tool_use_to_gemini(p))
                        case ReasoningPart():
                            # Reasoning parts are internal to the model, skip
                            pass

                if gemini_parts:
                    gemini_contents.append(genai_types.Content(role="model", parts=gemini_parts))

    return _merge_consecutive_same_role_contents(gemini_contents)
