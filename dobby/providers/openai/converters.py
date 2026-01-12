"""Internal converters for OpenAI message formats."""

from openai.types.responses import (
    ResponseInputFileContentParam,
    ResponseInputImageContentParam,
    ResponseInputTextContentParam,
)

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

OpenAIContentPart = (
    ResponseInputTextContentParam | ResponseInputImageContentParam | ResponseInputFileContentParam
)


def _text_to_openai(part: TextPart) -> ResponseInputTextContentParam:
    """Convert TextPart to OpenAI input_text format."""
    return ResponseInputTextContentParam(type="input_text", text=part.text)


def _image_to_openai(part: ImagePart) -> ResponseInputImageContentParam:
    """Convert ImagePart to OpenAI input_image format."""
    match part.source:
        case URLImageSource(url=url):
            return ResponseInputImageContentParam(type="input_image", image_url=url)
        case Base64ImageSource(data=data, media_type=mt):
            return ResponseInputImageContentParam(
                type="input_image", image_url=f"data:{mt};base64,{data}"
            )
    raise ValueError(f"Unknown image source type: {part.source}")


def _document_to_openai(part: DocumentPart) -> ResponseInputFileContentParam:
    """Convert DocumentPart to OpenAI input_file format."""
    match part.source:
        case URLSource(url=url):
            return ResponseInputFileContentParam(type="input_file", file_url=url)
        case Base64PDFSource(data=data, media_type=mt):
            return ResponseInputFileContentParam(
                type="input_file",
                filename=part.filename,
                file_data=f"data:{mt};base64,{data}",
            )
        case PlainTextSource(data=data):
            return ResponseInputFileContentParam(
                type="input_file",
                filename=part.filename,
                file_data=data,
            )
        case FileDocumentSource(file_id=fid):
            return ResponseInputFileContentParam(type="input_file", file_id=fid)
    raise ValueError(f"Unknown document source type: {part.source}")


def content_part_to_openai(
    part: TextPart | ImagePart | DocumentPart,
) -> OpenAIContentPart:
    """Convert any content part to OpenAI format."""
    match part:
        case TextPart():
            return _text_to_openai(part)
        case ImagePart():
            return _image_to_openai(part)
        case DocumentPart():
            return _document_to_openai(part)
    raise ValueError(f"Unknown content part type: {part}")
