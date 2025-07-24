from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field
from langchain_core.outputs import GenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from new_llm import NewLLM
from user import User


class MyLLM(LLM):
    """
    使用Langchain封装NewLLM的大模型类
    """
    user: User = Field(..., description="The user associated with the LLM instance")

    def __init__(self, user: User, *args: Any, **kwargs: Any):
        super().__init__(user=user, *args, **kwargs)

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
        my_model = NewLLM()
        result = my_model.llm(question=prompt, user_id=self.user.user_id, need2store=True)
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
        my_model = NewLLM()
        result = my_model.llm(question=prompt)
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
    user = User(user_id='chatcmpl-dbb9df84524d44a6b4dac92235a922ac')
    myLLM = MyLLM(user=user)
    # print(myLLM)
    # result = myLLM.invoke('你好！')
    # result = myLLM.batch(['你是谁？', '我是小明'])
    # print(result)
    # for token in myLLM.stream("你好"):
    #     print(token, end="|", flush=True)

    model = myLLM

    prompt = ChatPromptTemplate.from_messages("你好！")

    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
