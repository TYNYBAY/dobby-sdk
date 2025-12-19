---
trigger: model_decision
description: Python Backend Coding Standards - FastAPI + SQLAlchemy
globs: "**/*.py"
---

# Python Backend Coding Standards

FastAPI backend with SQLAlchemy 2.0, Pydantic, and Ruff linter. Follow these conventions for all Python code.

## Key Principles
- Python 3.12+ with type hints everywhere
- Google-style docstrings
- Ruff for linting and formatting
- Repository pattern for database access
- Service classes for business logic

---

## Technology Stack

- Python 3.12+
- FastAPI (latest stable)
- SQLAlchemy 2.0 with async support
- Pydantic and pydantic-settings for validation/config
- Alembic for migrations
- Ruff for linting/formatting
- uv for package management
- Loguru for logging
- Argon2 for password hashing
- python-jose for JWT tokens
- asyncpg for PostgreSQL async driver

---

## Folder Structure

server/
├── main.py                     # FastAPI app entry point with lifespan
├── pyproject.toml              # Project dependencies (uv)
├── ruff.toml                   # Linter configuration
├── alembic/                    # Database migrations
├── src/
│   ├── api/                    # API layer (routes, schemas, services)
│   │   ├── __init__.py         # Router aggregation
│   │   └── {feature}/          # Feature module
│   │       ├── __init__.py
│   │       ├── router.py       # FastAPI routes
│   │       ├── schemas.py      # Pydantic models
│   │       ├── service.py      # Business logic
│   │       └── dependencies.py # Route dependencies (optional)
│   ├── core/                   # Application core
│   │   ├── config.py           # Settings via pydantic-settings
│   │   ├── exceptions.py       # Custom HTTP exceptions
│   │   ├── security.py         # JWT and password utilities
│   │   └── logger.py           # Loguru configuration
│   └── lib/                    # Shared libraries
│       ├── db/                 # Database layer
│       │   ├── base.py         # SQLAlchemy Base and mixins
│       │   ├── session.py      # Async session management
│       │   ├── models/         # SQLAlchemy models
│       │   ├── repositories/   # Data access layer
│       │   ├── enums.py        # Database enums
│       │   └── types.py        # Custom types (UUID7)
│       └── {domain}/           # Domain-specific libraries
└── tests/                      # Test files

---

## File Conventions

### Module Docstrings
Every Python file starts with a docstring describing its purpose:

"""User model for authentication and ownership.

Users can be regular (password-authenticated) or guests.
Each user owns chats and their messages.
"""

### Router Files
Routes list endpoints in the module docstring:

"""Chat streaming API routes.

Endpoints:
- POST /chat/stream - Stream chat response from LLM via SSE
"""

from fastapi import APIRouter, Depends

router = APIRouter(prefix="/chat", tags=["chat-stream"])

@router.post("/stream")
async def chat_stream(
    request: ChatStreamRequest,
    user: User = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Stream chat response from LLM.

    Delegates to ChatService for:
    1. Save user message to DB
    2. Get chat history
    3. Stream LLM response
    4. Save assistant message to DB
    """
    service = ChatService(db)
    return StreamingResponse(...)

---

## Docstring Format (Google Style)

Use Google-style docstrings for all functions and classes:

def process_data(input_data: str, max_length: int = 100) -> dict[str, Any]:
    """Process input data and return results.

    Args:
        input_data: Raw input string to process.
        max_length: Maximum allowed length (default 100).

    Returns:
        Dictionary containing processed results with keys:
        - status: Processing status
        - data: Processed content

    Raises:
        ValueError: If input_data is empty.
    """

For classes, document attributes in the class docstring:

class User(Base, TimestampMixin):
    """User account model.

    Attributes:
        id: Unique UUID7 identifier
        email: User's email address (unique, indexed)
        password: Bcrypt hashed password (nullable for guest users)
        is_active: Whether the account is enabled
        user_type: Account type (regular or guest)
        chats: User's chat sessions (relationship)
    """

---

## Pydantic Schemas

Use Field with alias for camelCase JSON serialization:

from pydantic import BaseModel, Field

class ChatStreamRequest(BaseModel):
    """Request body for chat streaming endpoint."""

    chat_id: UUID = Field(alias="chatId")
    message: ChatMessage = Field(alias="message")
    chat_model_config: ChatModelConfig = Field(alias="chatModelConfig")


class MessagePart(BaseModel):
    """A part of a message - text or file attachment."""

    type: Literal["text", "file"]
    text: str | None = None
    url: str | None = None

---

## SQLAlchemy Models

Use SQLAlchemy 2.0 style with mapped_column and Mapped type hints:

from sqlalchemy import String, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=generate_uuid7,
    )
    email: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    # Relationships
    chats: Mapped[list["Chat"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"

---

## Repository Pattern

Database access through repository classes with static methods:

class MessageRepository:
    """Repository for message persistence."""

    @staticmethod
    async def get_by_chat_id(db: AsyncSession, chat_id: UUID) -> list[Message]:
        """Get all messages for a chat.

        Args:
            db: Async database session.
            chat_id: Chat UUID to filter by.

        Returns:
            List of messages ordered by creation time.
        """
        result = await db.execute(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at)
        )
        return list(result.scalars().all())

    @staticmethod
    async def save(
        db: AsyncSession,
        chat_id: UUID,
        parent_id: UUID,
        role: str,
        parts: list[dict],
    ) -> Message:
        """Save a new message."""
        message = Message(
            chat_id=chat_id,
            parent_msg_id=parent_id,
            role=role,
            parts=parts,
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)
        return message

---

## Service Classes

Business logic in service classes initialized with database session:

class ChatService:
    """Chat business logic with agentic tool execution."""

    def __init__(self, db: AsyncSession):
        """Initialize chat service.

        Args:
            db: Async database session.
        """
        self.db = db

    async def stream_response(
        self,
        chat_id: UUID,
        user_id: UUID,
        user_message: ChatMessage,
        model_config: ChatModelConfig,
    ) -> AsyncIterator[str]:
        """Process user message with agentic tool execution loop.

        Args:
            chat_id: Chat UUID.
            user_id: User UUID.
            user_message: User's chat message.
            model_config: Model configuration.

        Yields:
            SSE formatted event strings.
        """
        ...

---

## Custom Exceptions

HTTP exceptions with consistent structure:

from fastapi import HTTPException, status

class AuthException(HTTPException):
    """Base exception for authentication errors."""

    def __init__(
        self,
        status_code: int = status.HTTP_401_UNAUTHORIZED,
        detail: str = "Authentication failed",
        headers: dict[str, str] | None = None,
    ) -> None:
        if headers is None:
            headers = {"WWW-Authenticate": "Bearer"}
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class InvalidCredentialsException(AuthException):
    """Raised when credentials are invalid."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

---

## Configuration

Use pydantic-settings for environment configuration:

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Environment name",
    )
    debug: bool = False
    database_url: str = "postgresql+asyncpg://localhost:5432/app"

settings = Settings()

---

## Type Hints

- Use modern Python 3.12+ syntax (list instead of List, dict instead of Dict)
- Use union syntax with pipe: str | None instead of Optional[str]
- Use Literal for fixed string values
- Use TYPE_CHECKING for circular import resolution

from typing import TYPE_CHECKING, Literal
from collections.abc import AsyncIterator

if TYPE_CHECKING:
    from .chat import Chat

def process(data: str | None = None) -> dict[str, Any]:
    ...

async def stream() -> AsyncIterator[str]:
    ...

---

## Commands

uv sync              # Install dependencies
uv run ruff check    # Check linting
uv run ruff format   # Format code
uv run alembic upgrade head  # Run migrations
uv add - uv remove   # to manage dependencies
---

## Checklist for New Features

- Create feature folder in src/api/{feature}/
- Add router.py with endpoint docstrings
- Add schemas.py with Pydantic models (camelCase aliases)
- Add service.py for business logic
- Add to src/api/__init__.py router aggregation
- Create models in src/lib/db/models/ if needed
- Create repository in src/lib/db/repositories/ if needed
- Add Alembic migration if schema changes
- Use Google-style docstrings with Args/Returns
- Add type hints to all functions