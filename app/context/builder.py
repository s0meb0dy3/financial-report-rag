from app.domain import ConversationState
from app.messages import BaseMessage, SystemMessage, UserMessage


class ContextBuilder:
    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    def build(self, state: ConversationState, new_user_text: str) -> list[BaseMessage]:
        history = list(state.messages)
        if self.system_prompt and not any(message.role == "system" for message in history):
            history.insert(0, SystemMessage(content=self.system_prompt))
        return [*history, UserMessage(content=new_user_text)]
