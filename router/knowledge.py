import os

from PIL import Image
from fastapi import APIRouter, Form, HTTPException

from mineruParse import MinerUParser
from sql_query import query

router = APIRouter(
    prefix="/qaapi/knowledge",
    tags=["知识库"]
)


def is_supported_file(file_path: str) -> bool:
    # 定义支持的文件扩展名
    supported_extensions = ('.pdf', '.jpg', '.jpeg', '.png')

    # 检查文件扩展名
    if not file_path.lower().endswith(supported_extensions):
        return False

    # 如果是图片，进一步检查文件内容是否有效
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png',)):
        try:
            with Image.open(file_path) as img:
                img.verify()  # 验证文件是否为有效图片
        except (IOError, SyntaxError, FileNotFoundError):
            return False

    return True


@router.post("/process_document")
async def process_document(input_file_path: str = Form(...), backend: str = Form(...)):
    try:
        # 检查输入路径是否存在
        if not os.path.exists(input_file_path):
            raise HTTPException(status_code=400, detail="Input file does not exist.")

        # 检查是否为文件
        if not os.path.isfile(input_file_path):
            raise HTTPException(status_code=400, detail="Input path is not a file.")

        # 检查文件类型是否支持
        if not is_supported_file(input_file_path):
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # 如果通过了所有检查，继续处理
        parser = MinerUParser(input_file_path=input_file_path)
        """To enable VLM mode, change the backend to 'vlm-xxx'"""
        # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-transformers", server_url="http://192.168.143.117:30000")  # more general.
        # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-sglang-engine", server_url="http://192.168.143.117:30000")  # faster(engine).
        result = parser.parse_doc(path_list=parser.doc_path_list, backend=backend)  # faster(client).
        return result
    except HTTPException as http_exc:
        # 如果前面已经抛出HTTPException，直接重新抛出（保持400）
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrival")
def retrival_knowledge_db(inputs: str = Form(...), number: int = Form(...)):
    try:
        if number <= 0:
            raise ValueError('Number must be a positive integer.')
        result = query(inputs, number=number)
        return result
    except ValueError as ve:
        # 捕获输入无效的异常
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        # 捕获其他异常
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")