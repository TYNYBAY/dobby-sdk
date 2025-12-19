from typing import Literal, TypedDict

__all__ = ["ImagePart", "ImageSource", "Base64ImageSource", "URLImageSource"]


class Base64ImageSource(TypedDict):

    type: Literal["base64"]

    data: str

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class URLImageSource(TypedDict):

    type: Literal["url"]

    url: str


type ImageSource = Base64ImageSource | URLImageSource


class ImagePart(TypedDict):

    type: Literal["image"]

    source: ImageSource
