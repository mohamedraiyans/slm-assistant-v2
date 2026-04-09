"""
Tests for MemoryService.

Covers: save, retrieve, ordering, clear, and empty-state behaviour.
"""

from app.services.memory_service import MemoryService, Message


def test_new_service_has_empty_history():
    # Arrange / Act
    memory = MemoryService()
    # Assert
    assert memory.get_all() == []


def test_save_single_message_is_stored():
    # Arrange
    memory = MemoryService()
    # Act
    memory.save("user", "hello")
    # Assert
    result = memory.get_all()
    assert len(result) == 1
    assert result[0].role == "user"
    assert result[0].content == "hello"


def test_save_preserves_insertion_order():
    # Arrange
    memory = MemoryService()
    # Act
    memory.save("user", "first")
    memory.save("assistant", "second")
    memory.save("user", "third")
    # Assert
    history = memory.get_all()
    assert [m.content for m in history] == ["first", "second", "third"]


def test_get_all_returns_copy_not_internal_list():
    # Ensure external mutation does not affect internal state
    # Arrange
    memory = MemoryService()
    memory.save("user", "hi")
    # Act
    copy = memory.get_all()
    copy.append(Message(role="user", content="injected"))
    # Assert
    assert len(memory.get_all()) == 1


def test_clear_removes_all_messages():
    # Arrange
    memory = MemoryService()
    memory.save("user", "one")
    memory.save("assistant", "two")
    # Act
    memory.clear()
    # Assert
    assert memory.get_all() == []


def test_clear_then_save_works_normally():
    # Arrange
    memory = MemoryService()
    memory.save("user", "before clear")
    memory.clear()
    # Act
    memory.save("user", "after clear")
    # Assert
    result = memory.get_all()
    assert len(result) == 1
    assert result[0].content == "after clear"
