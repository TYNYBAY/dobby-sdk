"""Internal converters for Anthropic message formats."""

from typing import Any

from ...types import (
    Base64ImageSource,
    Base64PDFSource,
    DocumentPart,
    FileDocumentSource,
    ImagePart,
    PlainTextSource,
    TextPart,
    URLImageSource,
    URLSource,
)

AnthropicContentBlock = dict[str, Any]


def _text_to_anthropic(part: TextPart) -> AnthropicContentBlock:
    """Convert TextPart to Anthropic text format."""
    return {"type": "text", "text": part.text}


def _image_to_anthropic(part: ImagePart) -> AnthropicContentBlock:
    """Convert ImagePart to Anthropic image format."""
    match part.source:
        case URLImageSource(url=url):
            return {"type": "image", "source": {"type": "url", "url": url}}
        case Base64ImageSource(data=data, media_type=mt):
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mt, "data": data},
            }
    raise ValueError(f"Unknown image source type: {part.source}")


def _document_to_anthropic(part: DocumentPart) -> AnthropicContentBlock:
    """Convert DocumentPart to Anthropic document format."""
    match part.source:
        case URLSource(url=url):
            return {"type": "document", "source": {"type": "url", "url": url}}
        case Base64PDFSource(data=data, media_type=mt):
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": mt, "data": data},
            }
        case PlainTextSource(data=data):
            return {"type": "text", "text": data}
        case FileDocumentSource():
            raise ValueError("FileDocumentSource is not supported by Anthropic")
    raise ValueError(f"Unknown document source type: {part.source}")


def content_part_to_anthropic(
    part: TextPart | ImagePart | DocumentPart,
) -> AnthropicContentBlock:
    """Convert any content part to Anthropic format."""
    match part:
        case TextPart():
            return _text_to_anthropic(part)
        case ImagePart():
            return _image_to_anthropic(part)
        case DocumentPart():
            return _document_to_anthropic(part)
    raise ValueError(f"Unknown content part type: {part}")
