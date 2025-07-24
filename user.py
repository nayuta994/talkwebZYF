import os
import uuid
import psycopg2
from PIL import Image

from psycopg2 import sql
from color_log import ColorLogger as logger
from docs_reader2split import split_tool
from embedding import embedding

logger = logger()


def is_supported_file(file_path: str) -> bool:
    # 定义支持的文件扩展名
    supported_extensions = ('.pdf', '.jpg', '.jpeg', '.png',)

    # 检查文件扩展名
    if not file_path.lower().endswith(supported_extensions):
        return False

    # 如果是图片，进一步检查文件内容是否有效
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            with Image.open(file_path) as img:
                img.verify()  # 验证文件是否为有效图片
        except (IOError, SyntaxError, FileNotFoundError):
            return False
    return True


class User:
    """
    用户类
    """

    def __init__(self, user_id: str = '', user_name: str = '', pw: str = '', file_path: str = '', file_analysis: bool = False):
        self._document_content_id = None  # 文档解析结果id，用于数据库查询
        # self.user_question: str = ''
        if user_id.strip() == '':
            self.user_id = str(uuid.uuid4())
        else:
            self.user_id = user_id
        self.user_name: str = user_name
        self.pw = pw
        self.conn_params = {
            "host": "192.168.143.117",
            "port": "9002",
            "database": "history_message",
            "user": "postgres",
            "password": "123456"
        }
        self.save_user(user_id, user_name, pw)

        self.file_path = file_path

    def process_document(self, backend: str = 'vlm-sglang-client'):
        from mineruParse import MinerUParser
        input_file_path: str = self.file_path
        try:
            # 检查输入路径是否存在
            if not os.path.exists(input_file_path):
                raise ValueError("Input file does not exist.")

            # 检查是否为文件
            if not os.path.isfile(input_file_path):
                raise ValueError("Input path is not a file.")

            # 检查文件类型是否支持
            if not is_supported_file(input_file_path):
                raise ValueError("Unsupported file type. Expect a pdf, jpg, jpeg, pon type")

            # 如果通过了所有检查，继续处理
            parser = MinerUParser(input_file_path=input_file_path)
            """To enable VLM mode, change the backend to 'vlm-xxx'"""
            # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-transformers", server_url="http://192.168.143.117:30000")  # more general.
            # parser.parse_doc(path_list=parser.doc_path_list, backend="vlm-sglang-engine", server_url="http://192.168.143.117:30000")  # faster(engine).
            result = parser.parse_doc(path_list=parser.doc_path_list, backend=backend)  # faster(client).
            content = list(result.values())[0]
            self._document_content_id = self.save_document(content)
            return content

        except Exception as e:
            # 处理所有类型的异常
            logger.warning(f"[ERROR] An error occurred: {e}")

    def get_history_session(self, rounds: int = 5) -> list[dict[str, str]]:
        """
        获取历史会话
        参数：
        rounds: int  对话轮数
        返回值：
        history_session: list[dict[str, str]]  历史消息列表[ {"Q": user_message, "A": llm_message} , ]，最大轮数为rounds
        """
        user_id = self.user_id
        conn_params = self.conn_params
        try:
            logger.info("[INFO] SQL Query: Query from the message_history table:",
                        f'id:{user_id}, username:{self.user_name}, pw:{self.pw}')
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    # 查询
                    sql_query = sql.SQL("""
                    SELECT user_message, llm_message 
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
        for user_message, llm_message in query_results:
            history_session.append({"Q": user_message, "A": llm_message})
        history_session = history_session[::-1]
        return history_session

    def save_user(self, user_id, user_name, pw):
        """
        保存用户信息到users表
        """
        conn_params = self.conn_params
        try:
            # 先写user表
            logger.info("[INFO] User table: Data to be stored:", f'id:{user_id}, username:{''}, pw:{''}')  # 打印待存储的数据
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    insert_query = sql.SQL("""
                            INSERT INTO users (id, username, pw)  
                            VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING;
                        """)  # 需要修改是否插入name和pw字段！！
                    cur.execute(insert_query, (user_id, user_name, pw))
                    conn.commit()
                    logger.info("[INFO] User table updated successfully.")  # 打印用户表更新成功日志

        except Exception as e:
            logger.info(f"[ERROR] An error occurred: {e}")  # 打印错误日志

    def save_document(self, content: str):
        """
               保存文档内容和向量化的向量到document_content表
        """
        splits = split_tool([content], 512, 100)  # 切分
        print(splits)
        data = embedding([split.page_content for split in splits])  # 嵌入
        data_list = list(data)
        conn_params = {
            "host": "192.168.143.117",
            "port": "9002",
            "database": "knowledge_database",
            "user": "postgres",
            "password": "123456"
        }
        try:
            # 写文档内容和向量化的向量到document_content表
            logger.info("[INFO] document_content table: Data to be stored:")  # 打印待存储的数据
            content_ids = []
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    insert_query = sql.SQL("""
                        INSERT INTO document_content (content, embedding) 
                        VALUES (%s, %s)
                        RETURNING id;
                        """)
                    for data in data_list:
                        cur.execute(insert_query, data)
                        content_ids.append(cur.fetchone()[0])
                    self._document_content_id = content_ids
                    logger.info(f"[INFO] document_content table updated {len(data_list)} lines successfully.")  # 打印用户表更新成功日志
            return content_ids  # 原文档内容别切分后在的数据库存储的id列表
        except Exception as e:
            logger.warning(f"[ERROR] An error occurred: {e}")  # 打印错误日志

    def query_document(self, inputs: str, p: float = 0.8):
        """
        根据用户输入在文档向量库里面检索相似的文档内容片段，默认是10条
        !!! 可能需要确定一个保留策略，即相关标准阈值，来缩减不必要的检索结果，以缩短大模型的输入
        """
        document_content_id = self.document_content_id
        if document_content_id is None:
            logger.warning("未进行文档处理！")
            return []
        question_after_embedding = list(embedding([inputs]))[0][1]
        number = int(len(document_content_id)*p)

        conn_params = {
            "host": "192.168.143.117",
            "port": "9002",
            "database": 'knowledge_database',
            "user": "postgres",
            "password": "123456"
        }
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # 查询
                sql_query = sql.SQL("""
                    SELECT content, embedding <-> %s::real[] as similarity
                    FROM document_content
                    where id in ({})
                    ORDER BY embedding <-> %s::real[]
                    LIMIT %s;
                """).format(sql.SQL(", ").join(map(sql.Literal, document_content_id)))
                cur.execute(sql_query, (question_after_embedding, question_after_embedding, number))
                results = cur.fetchall()
                # for row in results:
                #     print(row)
        return results  # 返回list[tuple[内容,similarity]]


    @property
    def document_content_id(self):
        return self._document_content_id


if __name__ == '__main__':
    user = User('123456', file_path='inputs/Chapter2社会网络的基础知识.pdf', file_analysis=True)

    # history = user.get_history_session(3)
    # print(history)
    # 处理文档
    content = user.process_document()
    print(content)
    # 获取向量库中文档片段的id，用于后续检索
    print(user.document_content_id)

    # 根据问题再文档片段内进行检索，百分比为p
    query_result = user.query_document("我想对社区的人群网络进行分析，该使用什么策略？", p=0.8)
    print(query_result)
