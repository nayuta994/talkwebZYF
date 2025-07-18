from langchain_community.chat_models import ChatZhipuAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
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
# template = "给我一个关于{topic}的故事"
# prompt = ChatPromptTemplate.from_template(template)
# output_parser = StrOutputParser()
# message = {
#     'topic':'猫和老鼠'
# }
# chain = prompt | model | output_parser
# print(chain.invoke(message))
os.environ["ZHIPUAI_API_KEY"] = "a7cef2e24ea7bbb175fd2e86e9030c11.ZSx6hlQiUJXFSQAu"
model = ChatZhipuAI(model="glm-4-flash")

# 创建一个来聊天提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个擅长{ability}的助手,回答不超过50字"
        ),
        # MessagesPlaceholder 存放的是你的历史聊天记录 ， 作为占位
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
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # 保存历史聊天记录
    return store[session_id]


# 创建一个带有历史会话记录的Runnable, 我们指定了 input_messages_key（要作为最新输入消息处理的键）和 history_messages_key（要添加历史消息的键）
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history'
)
# 调用带有历史会话记录的Runnable来执行任务 ， 其中我们需要传递本次会话的session_id值
print(with_message_history.invoke(
    input={"ability":"数学","input":"正弦函数是什么意思"},
    config={"configurable" : { "session_id" : "session01" }}
))
