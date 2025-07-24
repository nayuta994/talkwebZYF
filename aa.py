from langchain_community.chat_models import ChatZhipuAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

import os

# openAI的API密钥
# os.environ["OPENAI_API_KEY"] = 'sk-dkmqf2YVlfXMIvEgGNVyGzoAMi8RKwgen994HiX90bqGIsZF'
# # 本地代理
# os.environ['http_proxy'] = '127.0.0.1:7897'
# os.environ['https_proxy'] = '127.0.0.1:7897'
# # 此处为LangSmith监测，可以忽略
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "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"

# prompt = ChatPromptTemplate.from_template(
#     "{text}.Please translate this sentence from {input_language} to {output_language}")
# model = ChatOpenAI(model="gpt-4o-mini")
# output_parser = StrOutputParser()
#
# chain = prompt | model | output_parser
# messages = {
#
#     "text": "生活不易，我学LangChain",
#     "input_language": "Chinese",
#     "output_language": "English",
# }
# print(chain.invoke(messages))


# model = ChatOpenAI(model="gpt-4o-mini")
# template = [
#     ('user', '{user}'),
#     ('human', "给我一个关于{topic}的故事")
# ]
# prompt = ChatPromptTemplate.from_messages(template)
# output_parser = StrOutputParser()
# message = {
#     'user':'老师',
#     'topic':'猫和老鼠'
# }
# chain = prompt | model | output_parser
# print(chain.invoke(message))

os.environ["ZHIPUAI_API_KEY"] = "a7cef2e24ea7bbb175fd2e86e9030c11.ZSx6hlQiUJXFSQAu"
model = ChatZhipuAI(model="glm-4-flash")

# 创建一个来聊天提示词模板
prompt = ChatPromptTemplate.from_messages(  # 注：from_messages不是from_template
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
# 定义一个通过session_id来获取会话记录的函数,得到的是一个ChatMessageHistory对象
# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()  # 保存历史聊天记录
#     return store[session_id]

# 这里我们需要传递两个参数来获取历史记录
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
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
# 创建一个带有历史会话记录的Runnable, 我们指定了 input_messages_key（要作为最新输入消息处理的键）和 history_messages_key（要添加历史消息的键）
# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key='input',
#     history_messages_key='history'
# )
# 调用带有历史会话记录的Runnable来执行任务 ， 其中我们需要传递本次会话需要的session_id值
message1 = with_message_history.invoke(
    input={"ability":"数学",'language':'中文',"input":"正弦函数是什么意思"},
    config={"configurable" : { 'user_id':'1',"conversation_id" : "session01" }}
)
print(message1)
message2 = with_message_history.invoke(
    input={"ability":"语文",'language':'中文', "input":"先输出上一次回答的内容，然后根据上一次的回答写诗"},
    config={"configurable" : {  'user_id':'1', "conversation_id" : "session01" }}  # 成功
)
print(message2)
print(store)
'''
{ ('1', 'session01'): InMemoryChatMessageHistory(
                                                messages=[ HumanMessage(content='正弦函数是什么意思', additional_kwargs={}, response_metadata={}), 
                                                AIMessage(content='正弦函数是一种数学函数，描述一个角度与直角三角形对边长度与斜边长度比例的关系。', additional_kwargs={}, response_metadata={}), 
                                                HumanMessage(content='先输出上一次回答的内容，然后根据上一次的回答写诗', additional_kwargs={}, response_metadata={}), 
                                                AIMessage(content='正弦函数是一种数学函数，描述一个角度与直角三角形对边长度与斜边长度比例的关系。\n\n正弦波荡，角度翩翩，\n三角函数，斜边比例间。', additional_kwargs={}, response_metadata={})
                                                    ]
                                                )
                                            }
'''   # InMemoryChatMessageHistory继承BaseChatMessageHistory


# message3 = with_message_history.invoke(
#     input={"ability":"语文","question":"先输出上一次回答的内容，然后根据上一次的回答写诗"},
#     config={"configurable" : { "session_id" : "session03" }}  # 没有保留之前的会话（id）
# )
# print(message3)

# os.environ["ZHIPUAI_API_KEY"] = "a7cef2e24ea7bbb175fd2e86e9030c11.ZSx6hlQiUJXFSQAu"
# model = ChatZhipuAI(model="glm-4-flash")
#
# store = {}
#
#
# def get_session_history(session_id):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]
#
#
# template = [
#     ('system', '你是一个{ability}专家'),
#     MessagesPlaceholder(variable_name='history_message'),
#     ('human', '{question}')
# ]
#
# prompt = ChatPromptTemplate.from_messages(template)
# out_parser = StrOutputParser()
# chain = prompt | model | out_parser
#
# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key='input',
#     history_messages_key='history_message'
# )
#
# # question = {
# #     'ability': '人工智能',
# #     'question': '使用ai解决肺栓塞诊断难题'
# # }
# response1 = with_message_history.invoke(
#     input={
#         'ability': '人工智能',
#         'question': '使用ai解决肺栓塞诊断难题'
#     },
#     config={'configurable': {'session_id': '001'}})
# print(response1)
#
# response2 = with_message_history.invoke(
#     input={
#         'ability': 'ai',
#         'question': '就前面讨论的问题进行可行性分析'
#     },
#     config={'configurable': {'session_id': '001'}})
# print(response2)

