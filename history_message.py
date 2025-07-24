from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

from user import User


class CustomChatMessageHistory(BaseChatMessageHistory):
    # messages: list[BaseMessage] = Field(default_factory=list)
    """A list of messages stored in memory."""
    def __init__(self, user: User, conversation_id: str = '', rounds: int = 3):
        self.user = user
        self.user_id = self.user.user_id
        self.conversation_id = conversation_id
        history_session = user.get_history_session(rounds)
        print(f'读取到历史消息的轮数：{len(history_session)}')
        self.messages = []
        for session in history_session:
            human_message = HumanMessage(content=session['Q'])
            ai_message = AIMessage(content=session['A'])
            self.messages.append(human_message)
            self.messages.append(ai_message)

        self.rounds = rounds

    def add_message(self, message: AIMessage|HumanMessage) -> None:
        self.messages.append(message)  # 替换为你的数据库插入逻辑

    def get_messages(self):
        return self.messages

    async def aget_messages(self) -> List[BaseMessage]:
        from langchain_core.runnables.config import run_in_executor
        return await run_in_executor(None, lambda: self.messages)

    def clear(self) -> None:
        self.messages = []