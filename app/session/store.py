from copy import deepcopy
from typing import Protocol

from app.domain import ConversationState


class SessionStore(Protocol):
    def load(self, session_id: str) -> ConversationState:
        ...

    def save(self, session_id: str, state: ConversationState) -> None:
        ...


class InMemorySessionStore:
    def __init__(self):
        self._states: dict[str, ConversationState] = {}

    def load(self, session_id: str) -> ConversationState:
        state = self._states.get(session_id)
        if state is None:
            return ConversationState()
        return deepcopy(state)

    def save(self, session_id: str, state: ConversationState) -> None:
        self._states[session_id] = deepcopy(state)
