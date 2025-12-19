from typing import Literal, TypedDict


class Base64PDFSource(TypedDict):

    type: Literal["base64"]

    data: str

    media_type: Literal["application/pdf"]


class PlainTextSource(TypedDict):

    type: Literal["text"]

    data: str

    media_type: Literal["text/plain"]


class URLSource(TypedDict):

    type: Literal["url"]

    url: str


class FileDocumentSource(TypedDict):

    type: Literal["file"]
    
    file_id: str
    

type DocumentSource = Base64PDFSource | PlainTextSource | URLSource | FileDocumentSource


class DocumentPart(TypedDict):

    type: Literal["document"]

    source: DocumentSource

    filename: str
