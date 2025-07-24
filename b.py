from my_llm import MyLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

from user import User
from history_message import CustomChatMessageHistory

if __name__ == '__main__':
    user = User(user_id='chatcmpl-dbb9df84524d44a6b4dac92235a922ac')
    myLLM = MyLLM(user=user)
    model = myLLM

    template = """
        ### 检索内容
        {retrieved_content}

        ### 历史消息
        {history}

        ### 用户输入
        {user_input}

        ### 指令
        请根据以上检索内容、历史消息和用户输入，生成合适的回答。
        确保回答准确、相关，并尽量使用检索内容中的信息。
        如果检索信息与用户输入相关性不大，可以指明并自己作答。
        如果用户输入与历史消息相关，请保持对话的连贯性。

    """

    prompt = ChatPromptTemplate.from_messages(
        [

            # MessagesPlaceholder 消息占位符--存放的是你的历史聊天记录或其他内容 ， 作为占位
            MessagesPlaceholder(variable_name='history'),
            (
                "system",
                "{ability}"
            ),
            (
                "human",
                "{input}"
            )
        ]
    )
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    store = {}


    def get_session_history(user: user):
        user_id = user.user_id
        if user_id not in store:
            store[user_id] = CustomChatMessageHistory(user, rounds=2)
        return store[user_id]


    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        # 定义一些自定义的参数
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user",
                annotation=User,
                name="用户示例",
                description="用户对象，其属性user_id为唯一标识符",
                default="",
                is_shared=True,
            ),
            # ConfigurableFieldSpec(
            #     id="conversation_id",
            #     annotation=str,
            #     name="对话 ID",
            #     description="对话的唯一标识符。",
            #     default="",
            #     is_shared=True,
            # ),
        ],
    )

    message1 = with_message_history.invoke(
        input={"ability": "文学", "input": "我养了一只小狗，它叫小白"},
        config={"configurable": {"user": user}}
    )
    ''' 存到数据库的user_message：-----然后下一次读取历史消息都堆加前面的消息！
    Human: 介绍一下华中科技大学
    AI: 华中科技大学

    Human: 讲一个故事
    AI: 好的，我将按照您的要求创作一个奇幻故事。

    System: 你是一个擅长文学的助手
    Human: 先输出前面的故事，然后续写前面的故事
    '''
    print(message1)
    print(store[user.user_id].get_messages())
