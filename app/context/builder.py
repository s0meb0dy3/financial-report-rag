from dataclasses import dataclass

from app.domain import ConversationState
from app.messages import BaseMessage, SystemMessage, UserMessage


@dataclass
class HistoryPolicy:
    def truncate(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return list(messages)


@dataclass
class MessageComposer:
    def compose(self, history: list[BaseMessage], new_user_text: str) -> list[BaseMessage]:
        return [*history, UserMessage(content=new_user_text)]


class ContextBuilder:
    def __init__(
        self,
        history_policy: HistoryPolicy | None = None,
        message_composer: MessageComposer | None = None,
        system_prompt: str = "",
    ):
        self.history_policy = history_policy or HistoryPolicy()
        self.message_composer = message_composer or MessageComposer()
        self.system_prompt = system_prompt

    def build(self, state: ConversationState, new_user_text: str) -> list[BaseMessage]:
        history = list(state.messages)
        if self.system_prompt and not any(message.role == "system" for message in history):
            history.insert(0, SystemMessage(content=self.system_prompt))
        history = self.history_policy.truncate(history)
        return self.message_composer.compose(history, new_user_text)
