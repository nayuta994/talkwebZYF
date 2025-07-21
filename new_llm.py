import json
import re
from typing import Generator, Any

import psycopg2
import requests
from psycopg2 import sql
from datetime import datetime
from color_log import ColorLogger as logger

logger = logger()


class NewLLM:
    def __init__(self):
        self.llm_answer: str = ''
        self.response: dict = None

    def _store_to_database(self, user_message: str, llm_message: str, user_id: str) -> None:
        """
                将用户消息和LLM回复存储到数据库中。
                参数:
                    user_message (str): 用户输入的消息。
                    llm_message (str): LLM生成的回复。
                    user_id (str): 用户唯一标识符。
                异常:
                    Exception: 如果数据库操作失败，记录错误日志。
                """
        conn_params = {
            "host": "192.168.143.117",
            "port": "9002",
            "database": "history_message",
            "user": "postgres",
            "password": "123456"
        }
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            # 先写user表
            logger.info("[INFO] User table: Data to be stored:", f'id:{user_id}, username:{''}, pw:{''}')  # 打印待存储的数据
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    insert_query = sql.SQL("""
                            INSERT INTO users (id, username, pw)  
                            VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING;
                        """)  # 需要修改是否插入name和pw字段！！
                    cur.execute(insert_query, (user_id, '', ''))
                    conn.commit()
                    logger.info("[INFO] User table updated successfully.")  # 打印用户表更新成功日志

            # 再写消息
            logger.info("[INFO] message_history table: Data to be stored:",
                        f'user_id:{user_id}, user_message:{user_message}, llm_message:{llm_message}, time:{time}')  # 打印待存储的数据
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    insert_query = sql.SQL("""
                            INSERT INTO message_history (user_id, user_message, llm_message, time)
                            VALUES (%s, %s, %s, %s)
                        """)
                    cur.execute(insert_query, (user_id, user_message.strip(), llm_message.strip(), time))
                    conn.commit()
                    logger.info("[INFO] Message history stored successfully.")

        except Exception as e:
            logger.info(f"[ERROR] An error occurred: {e}")  # 打印错误日志

    def _stream_response(self, response: list[str]) -> Generator[str]:
        """
        流式处理LLM的响应，逐块生成内容。
        参数:
            response (list): 来自LLM服务的流式响应读取得到的列表。
        返回:
            generator: 生成器，逐块产生LLM的回复内容。
        """
        # 流式处理
        full_response = ''
        for chunk in response:
            # response.iter_content返回一个生成器，每次 yield 一个字节块（bytes 类型，除非 decode_unicode=True）
            # decode_unicode=True-->会自动将字节流解码为 Unicode 字符串
            if chunk:  # 过滤掉 keep-alive 的空 chunk------type:str
                try:
                    # chunk:data: {"id":"chatcmpl-6f3e81b4cde246e98214effdf474737a",
                    # "object":"chat.completion.chunk","created":1753060607,"model":"qwen3-14b-fp8",
                    # "choices":[{"index":0,"delta":{"content":"中文"},"logprobs":null,"finish_reason":null}]}

                    chunk_json = json.loads(chunk[5:])  # 解析为json方便提取内容
                    if "choices" in chunk_json and chunk_json["choices"]:
                        delta = chunk_json["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content  # 使用yield返回生成器
                            full_response += content
                except json.JSONDecodeError:
                    continue  # 忽略无法解析的片段

    def llm(self, question: str, user_id: str = '', prompt: str = '', need2store: bool = False,
            temperature: float = 0.3, top_p: float = 0.4, stream: bool = False) -> Generator[str] | str:
        """
                调用LLM服务获取回复，并可选择存储对话历史。
                参数:
                    question (str): 用户提出的问题。
                    prompt (str): 提示词，默认无提示词。
                    stream (bool): 是否使用流式响应，默认为False。
                    need2store (bool): 是否需要存储对话历史，默认为False。
                    user_id (str): 用户唯一标识符，默认为调用大模型产生的随机id。
                返回:
                    stream=True时即流式输出：返回Generator[str]: 生成器，逐行产生LLM的回复内容。
                    stream=False时即非流式输出：返回str: 字符串，即大模型输出内容。
                异常:
                    requests.exceptions.RequestException: 如果LLM请求失败。
                    KeyError: 如果解析LLM响应时出错。
                    Exception: 其他未知错误。
                """
        url = 'http://192.168.143.117:9212/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer gpustack_0bcd33ceb0c8f708_e5309686ce88e01dbae423bb086f180f'
        }
        data = {
            "model": "qwen3-14b-fp8",
            "messages": [
                {
                    'role': 'system',
                    'content': prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        try:
            # 发送请求到LLM服务
            response = requests.post(url, headers=headers, data=json.dumps(data), stream=stream)
            response.raise_for_status()  # 检查 HTTP 错误
            if not stream:
                # 非流式处理
                resp_json = response.json()
                resp = resp_json['choices'][0]['message']['content']

                if user_id == '':
                    user_id = resp_json['id']

                user_message = question
                llm_message = re.match('<think>.*?</think>(?P<answer>.*)', resp, re.S).group('answer')  # 正则剔除思考内容
                self.llm_answer = llm_message
                resp_json['user_message'] = user_message
                self.response = resp_json

                if need2store:
                    # 存储用户信息和对话到数据库
                    self._store_to_database(user_message, llm_message, user_id)
                return resp
            else:
                # 流式处理
                response_iter = response.iter_lines(decode_unicode=True)
                response_integrated_json = {
                    "id": None,
                    "object": None,
                    "model": None,
                    "created": None,
                    "choices": {"content": "", },
                    "usage": {},
                    "user_message": None
                }
                iter_lines_list = list(response_iter)
                n = 0
                for line in iter_lines_list:
                    n += 1
                    if line:
                        try:
                            json_data = json.loads(line[5:])
                        except json.JSONDecodeError:
                            continue  # 跳过无法解析为json的行
                        # print(f'No.{n}: {json_data}')
                        if len(json_data["choices"]) > 0:
                            response_integrated_json["choices"]["content"] += json_data["choices"][0]["delta"][
                                "content"]
                        else:
                            response_integrated_json["usage"] = json_data["usage"]
                            response_integrated_json["id"] = json_data["id"]
                            response_integrated_json["object"] = json_data["object"] + 's'
                            response_integrated_json["created"] = json_data["created"]
                            response_integrated_json["model"] = json_data["model"]
                            response_integrated_json["id"] = json_data["id"]
                            if user_id == '':
                                user_id = json_data["id"]
                            response_integrated_json["usage"] = json_data["usage"]
                            response_integrated_json["user_message"] = question

                content = response_integrated_json["choices"]["content"]
                llm_message = re.match('<think>.*?</think>(?P<answer>.*)', content, re.S).group('answer')  # 正则剔除思考内容
                self.llm_answer = llm_message
                self.response = response_integrated_json

                if need2store:
                    # 存储用户信息和对话到数据库
                    user_message = question
                    self._store_to_database(user_message, llm_message, user_id)

                return self._stream_response(iter_lines_list)  # 返回生成器

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM请求失败: {e}")
            raise

        except KeyError as e:
            logger.error(f"解析LLM响应时出错: {e}")
            raise

        except Exception as e:
            logger.error(f"未知错误: {e}")
            raise


if __name__ == '__main__':
    llm = NewLLM()
    resp = llm.llm('介绍一下毛主席', stream=True, need2store=True)
    for i in resp:
        print(i, end='')
    print(llm.response)
