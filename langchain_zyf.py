from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import BaseMessage

from my_llm import MyLLM
from new_llm import NewLLM
from langchain_core.outputs import GenerationChunk
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

from user import User

if __name__ == "__main__":
    user = User(user_id='chatcmpl-dbb9df84524d44a6b4dac92235a922ac')
    myLLM = MyLLM(user=user)
    model = myLLM

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个擅长{ability}的助手,使用{language}回答不超过50字"
            ),
            # MessagesPlaceholder 消息占位符--存放的是你的历史聊天记录 ， 作为占位
            MessagesPlaceholder(variable_name='history'),
            (
                "human",
                "{input}"
            )
        ]
    )
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    store = {}


    def get_session_history(user: user, conversation_id: str) -> BaseChatMessageHistory:
        # 这里我们需要传递两个参数来获取历史记录
        user_id = user.user_id
        if (user_id, conversation_id) not in store:
            store[(user_id, conversation_id)] = ChatMessageHistory()
        return store[(user_id, conversation_id)]


    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        # 定义一些自定义的参数
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="用户 ID",
                description="用户的唯一标识符。",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="对话 ID",
                description="对话的唯一标识符。",
                default="",
                is_shared=True,
            ),
        ],
    )

    # 调用带有历史会话记录的Runnable来执行任务 ， 其中我们需要传递本次会话需要的session_id值
    message1 = with_message_history.invoke(
        input={"ability": "数学", 'language': '中文', "input": "正弦函数是什么意思"},
        config={"configurable": {'user_id': '1', "conversation_id": "1"}}
    )
    print(message1)
    message2 = with_message_history.invoke(
        input={"ability": "语文", 'language': '中文', "input": "先输出上一次回答的内容，然后根据上一次的回答写诗"},
        config={"configurable": {'user_id': '1', "conversation_id": "1"}}
    )
    print(message2)

    # 先预设一些历史的消息
    # temp_chat_history = ChatMessageHistory()
    # temp_chat_history.add_user_message("我叫小明, 你好")
    # temp_chat_history.add_ai_message("你好")
    # temp_chat_history.add_user_message("我今天心情不错")
    # temp_chat_history.add_ai_message("祝你天天都有好心情")
    # temp_chat_history.add_user_message("我下午在踢足球")
    # temp_chat_history.add_ai_message(
    #     "这个下午你在踢足球，这听起来真有意思！希望你们玩得开心，也期待下一次一起踢足球的时刻~")
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "你是一个乐于助人的助手。尽力回答所有问题。提供的聊天历史包括与您交谈的用户的事实。"
    #         ),
    #         MessagesPlaceholder(variable_name='chat_history'),
    #         (
    #             "human",
    #             "{input}"
    #         )
    #     ]
    # )
    # output_parser = StrOutputParser()
    # chain = prompt | model | output_parser
    #
    # # 只保留最后的两条聊天记录
    # def trim_messages(chain_input):
    #     stored_message = temp_chat_history.messages
    #     if len(stored_message) <= 2:
    #         return False
    #     temp_chat_history.clear()
    #     for message in stored_message[-2:]:
    #         temp_chat_history.add_message(message)
    #     return True
    #
    #
    # chain_with_message_history = RunnableWithMessageHistory(
    #     chain,
    #     lambda session_id: temp_chat_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    # )
    # # 在调用时先执行trim_messages函数来进行消息裁剪
    # chain_with_trimming = (
    #         RunnablePassthrough.assign(message_trimmed=trim_messages)
    #         | chain_with_message_history
    # )
