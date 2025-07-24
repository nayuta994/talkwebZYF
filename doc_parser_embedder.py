from typing import List, Tuple, Dict, Any

from fastapi import FastAPI, Form, HTTPException
import nest_asyncio

from docs_reader2split import split_tool
from embedding import embedding, save_2_pgvector
from mineruParse import MinerUParser

nest_asyncio.apply()


class DocumentProcessor:
    def __init__(self):
        """初始化 DocumentProcessor 类。

            参数:
                无
        """
        self.result_parse_json = None
        self.parser = None
        self.extracted_text = None

    def parse_document(self, input_dir_name: str, backend="pipeline") -> list[
        tuple[str, str | list[str | dict[str, Any]] | None]]:
        """解析文档并返回结构化数据。

        参数:
            input_dir_name (str): 输入文档的目录名。
            output_dir_name (str): 输出结果的目录名。
            backend (str, 可选): 解析后端，默认为 "vlm-sglang-client"。

        返回:
            tuple[list[Any], dict[Any, Any]]: 一个元组，包含两个元素：
                - outputs_dir_path (list[Any]): 输出结果文件夹路径列表。
                - outputs_content_list_json_dict (dict[Any, Any]): 解析后的内容字典。
        """
        self.parser = MinerUParser(input_dir_name=input_dir_name)
        self.result_parse_json = self.parser.parse_doc(path_list=self.parser.doc_path_list, backend=backend,
                                                       server_url="http://192.168.143.117:30000")  # faster(client).
        # print(self.parser.doc_path_list,
        #       # [WindowsPath('D:/PycharmProjects/pgvector/inputs/EViews在经济计量学中的应用.pdf'), .....]
        #       self.parser.file_name_list)  # ['EViews在经济计量学中的应用', '可视化平台', '肺栓塞中期报告', '肺栓塞中期检查表']

        # outputs_dir_path = []
        # outputs_content_list_json_path = []
        # outputs_content_list_json_dict = {}
        # if backend == "pipeline":
        #     parse_method = "pipeline"
        #     # f"{pdf_file_name}_content_list.json"
        #     for pdf_file_name in self.parser.file_name_list:
        #         outputs_dir_path.append(prepare_env(output_dir_name, pdf_file_name, parse_method)[
        #                                     1])  # 输出结果文件夹路径['outputs\\可视化平台\\vlm', 'outputs\\肺栓塞中期报告\\vlm', 'outputs\\肺栓塞中期检查表\\vlm']
        #
        #         content_list_json_path = os.path.join(prepare_env(output_dir_name, pdf_file_name, parse_method)[1],
        #                                               f"{pdf_file_name}_content_list.json")
        #         outputs_content_list_json_path.append(
        #             content_list_json_path)  # 输出的json文件路径['outputs\\可视化平台\\vlm\\可视化平台_content_list.json', 'outputs\\肺栓塞中期报告\\vlm\\肺栓塞中期报告_content_list.json', 'outputs\\肺栓塞中期检查表\\vlm\\肺栓塞中期检查表_content_list.json']
        #         with open(content_list_json_path, 'r', encoding='utf-8') as file:
        #             data = json.load(file)
        #         outputs_content_list_json_dict[pdf_file_name] = data
        # else:
        #     parse_method = "vlm"
        #     for pdf_file_name in self.parser.file_name_list:
        #         outputs_dir_path.append(prepare_env(output_dir_name, pdf_file_name, parse_method)[
        #                                     1])  # 输出结果文件夹路径['outputs\\可视化平台\\vlm', 'outputs\\肺栓塞中期报告\\vlm', 'outputs\\肺栓塞中期检查表\\vlm']
        #
        #         content_list_json_path = os.path.join(prepare_env(output_dir_name, pdf_file_name, parse_method)[1],
        #                                               f"{pdf_file_name}_content_list.json")
        #         outputs_content_list_json_path.append(
        #             content_list_json_path)  # 输出的json文件路径['outputs\\可视化平台\\vlm\\可视化平台_content_list.json', 'outputs\\肺栓塞中期报告\\vlm\\肺栓塞中期报告_content_list.json', 'outputs\\肺栓塞中期检查表\\vlm\\肺栓塞中期检查表_content_list.json']
        #         with open(content_list_json_path, 'r', encoding='utf-8') as file:
        #             data = json.load(file)
        #         outputs_content_list_json_dict[pdf_file_name] = data
        # print(outputs_dir_path)
        # print(outputs_content_list_json_path)
        # print(outputs_content_list_json_dict)
        return self.result_parse_json

    def extract_text(self, result_parse_json_list) -> list[dict[Any, str]]:
        """从解析后的数据中提取文本。

        参数:
            outputs_dir_path (list[str]): 输出结果文件夹路径列表。
            parsed_content_list_json_data (dict): 解析后的内容字典。

        返回:
            list[dict[str, str]]: 包含文本数据的字典列表。
        """
        text_data = []
        for file_name, data_list in result_parse_json_list:
            text_dict = {file_name: ''}
            for data in data_list:
                if data['type'] == 'text':
                    text_dict[file_name] += data['text']
            text_data.append(text_dict)
        return text_data

    def extract_images(self):
        """从解析后的数据中提取图像"""
        pass

    def extract_metadata(self):
        """从解析后的数据中提取元数据"""
        pass

    def embed_text(self, text_list: list[dict]) -> dict[Any, Any] | dict[str, Any] | dict[str, str] | dict[
        bytes, bytes]:
        """将文本先分割后做嵌入。

            参数:
                text_list (list[dict]): 包含文本数据的字典列表。

            返回:
                包含切分后的字符串和embedding编码
        """
        texts = [''.join(list(text.values())) for text in text_list]
        # print(texts)

        split_documents = split_tool(texts, 6300, 200)  # split
        split_text = [t.page_content.strip() for t in split_documents]
        # print(split_text)
        result = embedding(split_text)  # embedding-维度是1024
        save_2_pgvector(list(result))
        return dict(result)

    def embed_images(self):
        """将图像嵌入到目标文档中。"""
        pass

    def process_document(self, input_dir_name: str, backend: str = 'pipeline') -> dict[Any, Any] | dict[str, Any] | \
                                                                                  dict[str, str] | dict[bytes, bytes]:
        """处理整个文档，包括解析和嵌入。

        参数:
            input_dir_name (str): 输入文档的目录名。
            output_dir_name (str): 输出结果的目录名。

        返回:
            无
        """
        parsed_data = self.parse_document(input_dir_name=input_dir_name, backend=backend)
        text_list = self.extract_text(parsed_data)  # [{name:text}, {},{}....]
        result_embedding = self.embed_text(text_list)
        return result_embedding


# --- FastAPI 应用 ---
app = FastAPI()


@app.post("/process/")
async def process_documents(input_dir: str = Form(...), backend: str = Form(...)):
    """处理文档的端点。

    参数:
        inputdir (str): 输入目录路径。
        backend (str): 后端类型。
        outputdir (str): 输出目录路径。

    返回:
        JSONResponse: 处理结果。
    """
    try:
        dp = DocumentProcessor()
        dp.process_document(input_dir, backend)
        return dp.result_parse_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    # dp = DocumentProcessor()
    # dp.process_document('inputs', 'vlm-sglang-client')
    # print(dp.result_parse_json)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
