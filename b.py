import copy
import json
import os
from pathlib import Path
from typing import List, Any
import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, Form, HTTPException
from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make


class MinerUParser:
    def __init__(self, input_dir_name: str, model_source: str = "modelscope"):
        # args
        self.output_dir_name = 'outputs'  # 只输出识别过程中产生的图片，保存在.\outputs\pdf_name\vlm(pipeline)\images里面
        self.file_name_list = None

        __dir__ = os.path.dirname(os.path.abspath(__file__))
        pdf_files_dir = os.path.join(__dir__, input_dir_name)
        # print(pdf_files_dir)
        if not os.path.isdir(pdf_files_dir):
            logger.error(f"输入目录不存在: {pdf_files_dir}")
            raise FileNotFoundError(f"输入目录不存在: {pdf_files_dir}")
        self.output_dir = os.path.join(__dir__, self.output_dir_name)
        pdf_suffixes = [".pdf"]
        image_suffixes = [".png", ".jpeg", ".jpg"]

        self.doc_path_list = []
        for doc_path in Path(pdf_files_dir).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes:
                self.doc_path_list.append(doc_path)

        """如果您由于网络问题无法下载模型，可以设置环境变量MINERU_MODEL_SOURCE为modelscope使用免代理仓库下载模型"""
        os.environ['MINERU_MODEL_SOURCE'] = model_source

    @staticmethod
    def __do_parse(output_dir,  # Output directory for storing parsing results
                   pdf_file_names: list[str],  # List of PDF file names to be parsed
                   pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
                   p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
                   backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
                   parse_method="auto",  # The method for parsing PDF, default is 'auto'
                   formula_enable=True,  # Enable formula parsing
                   table_enable=True,  # Enable table parsing
                   server_url=None,  # Server URL for vlm-sglang-client backend
                   # f_draw_layout_bbox=False,  # Whether to draw layout bounding boxes
                   # f_draw_span_bbox=False,  # Whether to draw span bounding boxes
                   # f_dump_md=False,  # Whether to dump markdown files
                   # f_dump_middle_json=False,  # Whether to dump middle JSON files
                   # f_dump_model_output=False,  # Whether to dump model output files
                   # f_dump_orig_pdf=False,  # Whether to dump original PDF files
                   # f_dump_content_list=True,  # Whether to dump content list files
                   # f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
                   start_page_id=0,  # Start page ID for parsing, default is 0
                   end_page_id=None,
                   # End page ID for parsing, default is None (parse all pages until the end of the document)
                   ):
        result_json = []
        if backend == "pipeline":
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
                pdf_bytes_list[idx] = new_pdf_bytes

            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=formula_enable,
                table_enable=table_enable)

            for idx, model_list in enumerate(infer_results):
                # model_json = copy.deepcopy(model_list)
                pdf_file_name = pdf_file_names[idx]
                local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
                image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

                images_list = all_image_lists[idx]
                pdf_doc = all_pdf_docs[idx]
                _lang = lang_list[idx]
                _ocr_enable = ocr_enabled_list[idx]
                middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang,
                                                             _ocr_enable, formula_enable)

                pdf_info = middle_json["pdf_info"]

                # pdf_bytes = pdf_bytes_list[idx]
                # if f_draw_layout_bbox:
                #     draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
                #
                # if f_draw_span_bbox:
                #     draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
                #
                # if f_dump_orig_pdf:
                #     md_writer.write(
                #         f"{pdf_file_name}_origin.pdf",
                #         pdf_bytes,
                #     )
                #
                # if f_dump_md:
                #     image_dir = str(os.path.basename(local_image_dir))
                #     md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
                #     md_writer.write_string(
                #         f"{pdf_file_name}.md",
                #         md_content_str,
                #     )

                # if f_dump_content_list:
                #     image_dir = str(os.path.basename(local_image_dir))
                #     content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                #     md_writer.write_string(
                #     f"{pdf_file_name}_content_list.json",
                #     json.dumps(content_list, ensure_ascii=False, indent=4),
                # )
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                result_json.append((f"{pdf_file_name}_content_list.json", content_list))
                # print(content_list)

                # if f_dump_middle_json:
                #     md_writer.write_string(
                #         f"{pdf_file_name}_middle.json",
                #         json.dumps(middle_json, ensure_ascii=False, indent=4),
                #     )
                #
                # if f_dump_model_output:
                #     md_writer.write_string(
                #         f"{pdf_file_name}_model.json",
                #         json.dumps(model_json, ensure_ascii=False, indent=4),
                #     )

                logger.info(f"local output dir is {local_md_dir}")
        else:
            if backend.startswith("vlm-"):
                backend = backend[4:]

            # f_draw_span_bbox = False
            parse_method = "vlm"
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                pdf_file_name = pdf_file_names[idx]
                pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
                local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
                image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
                middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend,
                                                            server_url=server_url)

                pdf_info = middle_json["pdf_info"]

                # if f_draw_layout_bbox:
                #     draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
                #
                # if f_draw_span_bbox:
                #     draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
                #
                # if f_dump_orig_pdf:
                #     md_writer.write(
                #         f"{pdf_file_name}_origin.pdf",
                #         pdf_bytes,
                #     )
                #
                # if f_dump_md:
                #     image_dir = str(os.path.basename(local_image_dir))
                #     md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                #     md_writer.write_string(
                #         f"{pdf_file_name}.md",
                #         md_content_str,
                #     )

                # if f_dump_content_list:
                #     image_dir = str(os.path.basename(local_image_dir))
                #     content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                #     md_writer.write_string(
                #     f"{pdf_file_name}_content_list.json",
                #     json.dumps(content_list, ensure_ascii=False, indent=4),
                # )
                image_dir = str(os.path.basename(local_image_dir))
                content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                result_json.append((f"{pdf_file_name}_content_list.json", content_list))
                # print(content_list)

                # if f_dump_middle_json:
                #     md_writer.write_string(
                #         f"{pdf_file_name}_middle.json",
                #         json.dumps(middle_json, ensure_ascii=False, indent=4),
                #     )
                #
                # if f_dump_model_output:
                #     model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                #     md_writer.write_string(
                #         f"{pdf_file_name}_model_output.txt",
                #         model_output,
                #     )

                logger.info(f"local output dir is {local_md_dir}")
        return result_json

    def parse_doc(self, path_list: List[Path], lang: str = "ch", backend: str = "pipeline", method: str = "auto",
                  server_url: str = "http://192.168.143.117:30000", start_page_id: int = 0, end_page_id: int = None):
        """
        Parse the documents and store the results in the output directory.

        Parameters:
        path_list: List of Paths to the documents to be parsed.
        lang: Language option, default is 'ch'.
        backend: Backend for parsing PDF, default is 'pipeline'.
        method: Method for parsing PDF, default is 'auto'.
        server_url: Server URL for vlm-sglang-client backend.
        start_page_id: Start page ID for parsing, default is 0.
        end_page_id: End page ID for parsing, default is None.
        """
        if len(path_list) == 0:
            logger.warning("没有可解析的文件")
            print("提示: 没有可解析的文件，请检查输入目录。")
            return None
        try:
            self.file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            for path in path_list:
                file_name = str(Path(path).stem)
                pdf_bytes = read_fn(path)
                self.file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
                lang_list.append(lang)
            result_parse_json = self.__do_parse(
                output_dir=self.output_dir,
                pdf_file_names=self.file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id
            )
            text_data = {}
            for file_name, data_list in result_parse_json:
                text_data[file_name] = ''
                for data in data_list:
                    if data['type'] == 'text':
                        text_data[file_name] += data['text']
            return text_data

        except Exception as e:
            logger.error(f"Failed to parse documents: {e}")
            raise  # Re-raise the exception for further handling


# --- FastAPI 应用 ---
app = FastAPI()
@app.post("/process/")
async def process_documents(input_dir: str = Form(...), backend: str = Form(...)):
    try:
        parser = MinerUParser(input_dir_name=input_dir)
        """To enable VLM mode, change the backend to 'vlm-xxx'"""
        # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-transformers", server_url="http://192.168.143.117:30000")  # more general.
        # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-sglang-engine", server_url="http://192.168.143.117:30000")  # faster(engine).
        result = parser.parse_doc(path_list=parser.doc_path_list, backend=backend)  # faster(client).
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example usage:
if __name__ == "__main__":
    # parser = MinerUParser(input_dir_name='inputs')
    # """To enable VLM mode, change the backend to 'vlm-xxx'"""
    # # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-transformers", server_url="http://192.168.143.117:30000")  # more general.
    # # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-sglang-engine", server_url="http://192.168.143.117:30000")  # faster(engine).
    # result = parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-sglang-client",
    #                           server_url="http://192.168.143.117:30000")  # faster(client).
    # # print(parser.doc_path_list)
    # print(result)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

