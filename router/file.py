import os
from typing import List

from PIL import Image
from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel

from user import User

router = APIRouter(
    prefix="/qaapi/file",
    tags=["解析用户上传文件"]
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


class ItemList(BaseModel):
    items: List[int]


@router.post("/process_file")
async def process_file(user_id: str = Form(...), file_path: str = Form(...)):
    try:
        # 检查输入路径是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Input file does not exist.")

        # 检查是否为文件
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail="Input path is not a file.")

        # 检查文件类型是否支持
        if not is_supported_file(file_path):
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        user = User(user_id=user_id, file_path=file_path, is_file_analysis=True)
        content = user.process_document()
        content_chunk_id_list = user.document_content_id
        result = {
            "content": content,
            "content_chunks_id_list": content_chunk_id_list
        }
        return result
    except HTTPException as http_exc:
        # 如果前面已经抛出HTTPException，直接重新抛出（保持400）
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

