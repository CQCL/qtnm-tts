from typing import ClassVar, Protocol


class IsDataclass(Protocol):
    """Protocol for checking if a class is a dataclass."""

    __dataclass_fields__: ClassVar[dict[str, str]]


CoeffType = int | float | complex
