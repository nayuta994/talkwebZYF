import os
from datetime import datetime
import psycopg2
from PIL import Image
from fastapi import APIRouter, Query, Form, HTTPException, status
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from psycopg2 import sql
from main import main
import user
from color_log import ColorLogger as logger
from my_llm import MyLLM
from user import User
from pydantic import BaseModel

router = APIRouter(
    prefix="/qaapi/chat",
    tags=["问答"]
)

logger = logger()

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


class Item(BaseModel):
    user_message: str
    llm_message: str
    time: datetime


@router.post("/communication")  # 对话
async def communicate(question: str = Form(...), user_id: str = Form(...), is_file_analysis: bool = Form(...), user_file_path: str = Form(...)):
    """
    对话
    """
    try:
        # 检查输入路径是否存在
        if not os.path.exists(user_file_path):
            raise HTTPException(status_code=400, detail="Input file does not exist.")

        # 检查是否为文件
        if not os.path.isfile(user_file_path):
            raise HTTPException(status_code=400, detail="Input path is not a file.")

        # 检查文件类型是否支持
        if not is_supported_file(user_file_path):
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        result = main(question, user_file_path, user_id,is_file_analysis)
        return result
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logger.error(f"Error in /communication: {str(e)}")
        # 返回 HTTP 500 内部服务器错误
        raise HTTPException(status_code=500, detail="Internal server error while processing your request.")


@router.get("/conversations/{user_id}")  # 查询对话历史
async def get_conversations(user_id: str,
                            rounds: str = Query('all', description='对话轮数')):
    # 验证 rounds 参数
    if rounds != 'all' and not rounds.isdigit():
        if int(rounds) <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid 'rounds' parameter. Must be 'all' or a positive integer.")

    try:
        user = User(user_id=user_id)
        user_id = user.user_id
        conn_params = user.conn_params
        try:
            logger.info("[INFO] SQL Query: Query from the message_history table:",
                        f'id:{user_id}, username:{user.user_name}, pw:{user.pw}')
            query_results = []  # 存放查询结果
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    # 查询
                    if rounds == 'all':
                        sql_query = sql.SQL("""
                                   SELECT id,user_message,llm_message,time
                                   FROM message_history
                                   WHERE user_id=%s
                                   ORDER BY time DESC;
                                   """)
                        cur.execute(sql_query, (user_id,))
                    else:
                        sql_query = sql.SQL("""
                                            SELECT id,user_message,llm_message,time
                                            FROM message_history
                                            WHERE user_id=%s
                                            ORDER BY time DESC
                                            LIMIT %s;
                                            """)
                        cur.execute(sql_query, (user_id, rounds))
                    query_results = cur.fetchall()
        except Exception as e:
            logger.warning(f"[ERROR] An error occurred: {e}")  # 打印错误日志
        history_session = []
        for id, user_message, llm_message, time in query_results:
            history_session.append({"id": id, "Q": user_message, "A": llm_message, "time": time})
        history_session = history_session[::-1]
        return history_session
    except (psycopg2.OperationalError, psycopg2.Error) as e:
        logger.warning(f"[ERROR] Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred.")
    except Exception as e:
        logger.warning(f"[ERROR] An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.put("/conversations/{id}")  # 修改历史对话内容
async def modify_conversation(id: int, item: Item):
    try:
        user.update_conversation(id, item.user_message, item.llm_message, item.time)
    except Exception as e:
        logger.info(f"[ERROR] An error occurred: {e}")  # 打印错误日志
    return {"message": "Item updated", "item": item.dict()}


@router.post("/process_file")
async def process_file(user_id: str = Form(...), file_path: str = Form(...)):  # 处理上传的文件
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
        document_content_id = user.document_content_id
        result = {
            "content": content,
            "document_content_id": document_content_id
        }
        return result
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logger.error(f"Error in /communication: {str(e)}")
        # 返回 HTTP 500 内部服务器错误
        raise HTTPException(status_code=500, detail="Internal server error while processing your request.")
