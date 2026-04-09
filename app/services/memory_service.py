"""
MemoryService — stores conversation history as a list of role/message pairs.

Intentionally kept simple: in-memory only, no persistence.
Extend with a database backend if multi-session history is needed.
"""

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


class MemoryService:
    """Maintains an ordered list of conversation messages."""

    def __init__(self):
        self._history: list = []

    def save(self, role: str, message: str) -> None:
        """Append a message to the conversation history."""
        self._history.append(Message(role=role, content=message))

    def get_all(self) -> list:
        """Return a copy of the full conversation history."""
        return list(self._history)

    def clear(self) -> None:
        """Wipe the conversation history."""
        self._history.clear()
