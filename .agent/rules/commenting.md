---
trigger: model_decision
description: Code Commenting Standards - Good vs Bad Practices
globs: "**/*.{py,ts,tsx,js,jsx}"
---

# Code Commenting Standards

Comments explain WHY, not WHAT. If the code is clear, no comment is needed.

---

## Good Practices

### Module Docstrings
Describe file purpose and key contents:

"""Chat service with agentic tool execution.

This module provides ChatService which handles user message processing,
LLM interaction with tool execution, and SSE streaming.
"""

### Function Docstrings
Document purpose, args, returns - not the implementation:

def create_speech_recognizer() -> tuple[Recognizer, AudioStream]:
    """Creates Azure Speech Recognizer for compressed WebSocket audio.

    Returns:
        tuple: (SpeechRecognizer, PushAudioInputStream)
    """

### Class Docstrings with Attributes

class User(Base):
    """User account model.

    Attributes:
        id: Unique UUID7 identifier
        email: Email address (unique, indexed)
        password: Hashed password (nullable for guests)
    """

### Explain Non-Obvious Technical Decisions

# AudioStreamContainerFormat.ANY handles audio/webm;codecs:opus,
# which is the format sent by modern browsers over WebSocket
audio_format = AudioStreamFormat(compressed_stream_format=ANY)

### Explain Edge Cases or Tricky Logic

# evt.result contains COMPLETE recognized text, not incremental,
# so we extract only the new portion since last callback
if recognizing_text.startswith(previous_text):
    incremental_text = recognizing_text[len(previous_text):]

### Type Alias Documentation

type StopReason = Literal["end_turn", "max_tokens", "tool_use"]
"""Why the model stopped: natural end, token limit, or tool invocation."""

---

## Bad Practices

### Restating Code
# add five to x
y = x + 5

### Obvious Actions
# Save to database
await db.commit()

# Get chat history
messages = await get_chat_history(chat_id)

### Numbered Step-by-Step
# 1. Fetch user
# 2. Validate input
# 3. Save to database
If each step is a clear function call, no comments needed.

### No Context
# Used later
x = 10

---

## Rule of Thumb

Before adding a comment, ask: "Does the code already say this?"

If YES -> no comment needed
If NO -> comment explains WHY, not WHAT