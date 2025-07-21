from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from llm import llm
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

import os


class MyLLM(LLM):
    """A custom chat model that echoes the first `n` characters of the input.
    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.
    Example:
        .. code-block:: python
            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # 调用实际的模型API
        response = self._call_model_api(prompt, stop, **kwargs)
        return response

    def _call_model_api(self, prompt: str, stop: List[str], **kwargs) -> str:
        # 这里调用实际的模型API
        result = llm(question=prompt)
        # 这是一个示例实现
        return result

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.
        This method should be overridden by subclasses that support streaming.
        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.
        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.
        Returns:
            An iterator of GenerationChunks.
        """
        result = llm(question=prompt)
        for char in result:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel,based on qwen3-14b-fp8.",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "CustomChatModel,based on qwen3-14b-fp8."


# 使用封装后的模型
if __name__ == "__main__":
    myLLM = MyLLM()
    # print(myLLM)
    # result = myLLM.invoke('你好！')
    # result = myLLM.batch(['你是谁？', '我是小明'])
    # print(result)
    # for token in myLLM.stream("你好"):
    #     print(token, end="|", flush=True)

    model = myLLM

    # 1. 链的实现
    # prompt = ChatPromptTemplate.from_template(
    #     '{text},请使用{language}翻译这段话。'
    # )
    #
    # output_parser = StrOutputParser()
    #
    # chain = prompt | model | output_parser
    # resp = chain.invoke({
    #     'text': '我爱你',
    #     'language': '法语'
    # })
    # print(resp)

    # 2. 创建一个来聊天提示词模板
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
