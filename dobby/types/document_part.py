from dataclasses import dataclass
from typing import Literal


@dataclass
class Base64PDFSource:
    """Document source from base64 PDF data."""

    data: str

    media_type: Literal["application/pdf"]

    kind: Literal["base64"] = "base64"


@dataclass
class PlainTextSource:
    """Document source from plain text."""

    data: str

    kind: Literal["text"] = "text"


@dataclass
class URLSource:
    """Document source from URL."""

    url: str

    kind: Literal["url"] = "url"


@dataclass
class FileDocumentSource:
    """Document source from file ID."""

    file_id: str

    kind: Literal["file"] = "file"


type DocumentSource = Base64PDFSource | PlainTextSource | URLSource | FileDocumentSource


@dataclass
class DocumentPart:
    """A document content part."""

    source: DocumentSource

    filename: str

    kind: Literal["document"] = "document"
