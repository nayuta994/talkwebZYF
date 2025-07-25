from langchain_core.output_parsers import StrOutputParser

from my_llm import MyLLM
from prompt_genetator import prompt_generator
from sql_query import query
from user import User


def main(user_inputs: str, user_file_path: str = 'inputs/Chapter2社会网络的基础知识.pdf', user_id: str = 'chatcmpl-dbb9df84524d44a6b4dac92235a922ac', is_file_analysis: bool = True):

    user = User(user_id=user_id, file_path=user_file_path, is_file_analysis=is_file_analysis)
    user.process_document()  # 处理文档

    model = MyLLM(user=user)

    # 生成提示词
    prompt = prompt_generator()

    # 获取历史会话
    history = user.get_history_session(2)

    # 检索
    retrival_result_dict = {}
    # 检索文档内容
    for document_chunk in user.query_document(user_inputs, 0.8):
        retrival_result_dict[document_chunk[0]] = document_chunk[1]
    # 检索知识库
    for knowledge in query(user_inputs, number=5):
        retrival_result_dict[knowledge[0]] = knowledge[1]
    sorted_retrival_result_dict = sorted(retrival_result_dict.items(), key=lambda item: item[1])
    # 排序后的检索内容列表list[str]
    sorted_retrival_result_list = [item[0] for item in sorted_retrival_result_dict]
    # print(f'检索信息：{sorted_retrival_result_list}')
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    result = chain.invoke({
        "retrival_result": sorted_retrival_result_list,
        "history": history,
        "input": user_inputs
    })
    # print(result)
    return result


if __name__ == '__main__':
    user_id = input("输入用户id：")
    user_inputs = input("输入问题：")
    user_file_path = input("输入附件路径：")
    print(main(user_inputs=user_inputs))



