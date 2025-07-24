from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_tool(texts_list: list, chunk_size, chunk_overlap):
    # 读取文档
    docs = []
    for text in texts_list:
        if str(text).strip() != '':
            # print(text[:100])
            doc = Document(page_content=text)
            docs.append(doc)

    # 切分 - 基于字符估算 token 数量
    # 假设 token/word 比例约为 1.3，估算 8192 tokens 约等于 6300 字符
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 估算的字符数
        chunk_overlap=chunk_overlap,  # 重叠字符数
        length_function=len  # 使用 len 函数计算字符数
    )
    splits = splitter.split_documents(docs)  # 现在 splits 中的每个 Document 对象的 page_content 就是你准备传递给 BGE-M3 的文本片段

    # print(f"切分成了 {len(splits)} 个片段")
    # for i, split in enumerate(splits):
    #     # 可以打印片段长度，观察是否接近预期
    #     print(f"片段 {i + 1} 长度: {len(split.page_content)} 字符")
    return splits


# if __name__ == '__main__':
#     # 加载网页
#     os.environ["USER_AGENT"] = "MyApp/1.0 (1662546235@qq.com@example.com)"
#     loader = WebBaseLoader(
#         web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
#         bs_kwargs=dict(
#             parse_only=bs4.SoupStrainer(class_=["post-header", "post-title", "post-content"])
#         )
#     )
#     docs = loader.load()
#
#     print(f"加载了 {len(docs)} 个文档")
#     # print(len(docs[0].page_content))  #43047
#     # docs =[Document('')]  # 切分成了 0 个片段
#
#     # 切分 - 基于字符估算 token 数量
#     # 假设 token/word 比例约为 1.3，估算 8192 tokens 约等于 6300 字符
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=6300,  # 估算的字符数
#         chunk_overlap=200,  # 重叠字符数
#         length_function=len  # 使用 len 函数计算字符数
#     )
#     splits = splitter.split_documents(docs)  # 现在 splits 中的每个 Document 对象的 page_content 就是你准备传递给 BGE-M3 的文本片段
#     print(f"切分成了 {len(splits)} 个片段")
#     for i, split in enumerate(splits):
#         # 可以打印片段长度，观察是否接近预期
#         print(f"片段 {i + 1} 长度: {len(split.page_content)} 字符")
#
#     # with open('content.txt', mode='a', encoding='utf-8') as f:
#     #     for split in splits:
#     #         f.write(split.page_content + "\n")  # 添加两个换行符作为片段间的分隔
#     # print("内容已成功写入到 content.txt 文件中。")
