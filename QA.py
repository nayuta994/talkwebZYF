from llm import llm
from sql_query import query


if __name__ == '__main__':
    prompt = ''
    question = "我想了解一些关于使用ai解决肺栓塞问题相关的内容"

    # 加入解析pdf功能
    # from mineruParse import MinerUParser
    # parser = MinerUParser(input_dir_name='input', output_dir_name="outputs")
    # """To enable VLM mode, change the backend to 'vlm-xxx'"""
    # # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-transformers")  # more general.
    # # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-sglang-engine")  # faster(engine).
    # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-sglang-client",
    #                  server_url="http://192.168.143.117:30000")  # faster(client).

    result = query(question)
    prompt = prompt + ('下面是通过检索向量数据库得到的与问题相匹配的前五个文档和余弦相似度（值越小越相似）'
                       f'No.1:内容：{result[0][0]}  余弦相似度:{result[0][1]}'
                       f'No.2:内容：{result[1][0]}  余弦相似度:{result[1][1]}'
                       f'No.3:内容：{result[2][0]}  余弦相似度:{result[2][1]}'
                       f'No.4:内容：{result[3][0]}  余弦相似度:{result[3][1]}'
                       f'No.5:内容：{result[4][0]}  余弦相似度:{result[4][1]}\n'
                       f'请参考检索内容，如果检索到的内容质量不高，请自行做出回答。'
                       f'使用中文做出回答，注意不要完全仿照检索内容！')
    print(prompt)
    print(llm(question=question, prompt=prompt))
