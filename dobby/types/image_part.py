from dataclasses import dataclass
from typing import Literal


@dataclass
class Base64ImageSource:
    """Image source from base64 data."""

    data: str

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

    kind: Literal["base64"] = "base64"


@dataclass
class URLImageSource:
    """Image source from URL."""

    url: str

    kind: Literal["url"] = "url"


type ImageSource = Base64ImageSource | URLImageSource


@dataclass
class ImagePart:
    """An image content part."""

    source: ImageSource

    kind: Literal["image"] = "image"
