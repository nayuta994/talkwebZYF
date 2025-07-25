from color_log import ColorLogger as logger
from fastapi import FastAPI

from router.chat import router as chat_router
from router.file import router as file_router
from router.knowledge import router as knowledge_router

logger = logger()

app = FastAPI()
app.include_router(chat_router)  # 问答
app.include_router(knowledge_router)  # 知识库
app.include_router(file_router)  # 文档解析

# # 定义请求体模型
# class Item(BaseModel):
#     user_message: str
#     llm_message: str
#     time: datetime
#
#
# def is_supported_file(file_path: str) -> bool:
#     # 定义支持的文件扩展名
#     supported_extensions = ('.pdf', '.jpg', '.jpeg', '.png')
#
#     # 检查文件扩展名
#     if not file_path.lower().endswith(supported_extensions):
#         return False
#
#     # 如果是图片，进一步检查文件内容是否有效
#     if file_path.lower().endswith(('.jpg', '.jpeg', '.png',)):
#         try:
#             with Image.open(file_path) as img:
#                 img.verify()  # 验证文件是否为有效图片
#         except (IOError, SyntaxError, FileNotFoundError):
#             return False
#
#     return True
#
#
#
#
# @app.post("/qaapi/file/process_document/")  # 处理文档
# async def process_document(input_file_path: str = Form(...), backend: str = Form(...)):
#     try:
#         # 检查输入路径是否存在
#         if not os.path.exists(input_file_path):
#             raise HTTPException(status_code=400, detail="Input file does not exist.")
#
#         # 检查是否为文件
#         if not os.path.isfile(input_file_path):
#             raise HTTPException(status_code=400, detail="Input path is not a file.")
#
#         # 检查文件类型是否支持
#         if not is_supported_file(input_file_path):
#             raise HTTPException(status_code=400, detail="Unsupported file type.")
#
#         # 如果通过了所有检查，继续处理
#         parser = MinerUParser(input_file_path=input_file_path)
#         """To enable VLM mode, change the backend to 'vlm-xxx'"""
#         # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-transformers", server_url="http://192.168.143.117:30000")  # more general.
#         # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-sglang-engine", server_url="http://192.168.143.117:30000")  # faster(engine).
#         result = parser.parse_doc(path_list=parser.doc_path_list, backend=backend)  # faster(client).
#         return result
#     except HTTPException as http_exc:
#         # 如果前面已经抛出HTTPException，直接重新抛出（保持400）
#         raise http_exc
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @app.post("/qaapi/chat/communication")  # 对话
# async def communicate(question: str = None):
#     """
#     对话
#     """
#     user = User()
#     model = MyLLM(user=user)
#     prompt = ChatPromptTemplate.from_template(
#         "你是一个问答助手，请根据用户的问题做详细的解答。以下是用户的问题: {question}.")
#     output_parser = StrOutputParser()
#     message = {
#         'question': question
#     }
#     chain = prompt | model | output_parser
#     response = chain.invoke(message)
#     return response
#
#
# @app.get("/qaapi/chat/conversations/{user_id}")  # 查询对话历史
# async def get_conversations(user_id: str,
#                             rounds: str = Query('all', description='对话轮数')):
#     user = User(user_id=user_id)
#     user_id = user.user_id
#     conn_params = user.conn_params
#     try:
#         logger.info("[INFO] SQL Query: Query from the message_history table:",
#                     f'id:{user_id}, username:{user.user_name}, pw:{user.pw}')
#         query_results = []  # 存放查询结果
#         with psycopg2.connect(**conn_params) as conn:
#             with conn.cursor() as cur:
#                 # 查询
#                 if rounds == 'all':
#                     sql_query = sql.SQL("""
#                                SELECT id,user_message,llm_message,time
#                                FROM message_history
#                                WHERE user_id=%s
#                                ORDER BY time DESC;
#                                """)
#                     cur.execute(sql_query, (user_id,))
#                 else:
#                     sql_query = sql.SQL("""
#                                         SELECT id,user_message,llm_message,time
#                                         FROM message_history
#                                         WHERE user_id=%s
#                                         ORDER BY time DESC
#                                         LIMIT %s;
#                                         """)
#                     cur.execute(sql_query, (user_id, rounds))
#                 query_results = cur.fetchall()
#     except Exception as e:
#         logger.warning(f"[ERROR] An error occurred: {e}")  # 打印错误日志
#     history_session = []
#     for id, user_message, llm_message, time in query_results:
#         history_session.append({"id": id, "Q": user_message, "A": llm_message, "time": time})
#     history_session = history_session[::-1]
#     return history_session
#
#
# @app.put("/qaapi/chat/conversations/{id}")
# async def update_user(id: int, item: Item):
#     conn_params = {
#         "host": "192.168.143.117",
#         "port": "9002",
#         "database": "history_message",
#         "user": "postgres",
#         "password": "123456"
#     }
#     try:
#         logger.info("[INFO] message_history table: Data to be updated:")
#         with psycopg2.connect(**conn_params) as conn:
#             with conn.cursor() as cur:
#                 insert_query = sql.SQL("""
#                         UPDATE message_history
#                         set user_message=%s, llm_message=%s, update_time=%s
#                         where id =%s;
#                     """)
#                 cur.execute(insert_query, (item.user_message, item.llm_message, item.time, id))
#                 conn.commit()
#                 logger.info("[INFO] Message history stored successfully.")
#     except Exception as e:
#         logger.info(f"[ERROR] An error occurred: {e}")  # 打印错误日志
#     return {"message": "Item updated", "item": item.dict()}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
