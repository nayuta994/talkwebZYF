import json
import re

import psycopg2
import requests
from psycopg2 import sql
from datetime import datetime


class LLM:
    def __init__(self):
        self.response = ''

    def input(self,question, user_id='', prompt='', need2store=False, temperature=0.3, top_p=0.4, stream=False):
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
        resp_json = requests.post(url, headers=headers, data=json.dumps(data), stream=stream).json()
        resp = resp_json['choices'][0]['message']['content']

        if need2store:  # 存储到后台数据库
            conn_params = {
                "host": "192.168.143.117",
                "port": "9002",
                "database": "history_message",
                "user": "postgres",
                "password": "123456"
            }

            if user_id == '':
                user_id = resp_json['id']    # json会返回'id': 'chatcmpl-341c93aa6beb40bfa5c564a830e5d059'
            user_message = question
            llm_message = resp
            # llm_message = re.match('<think>.*?</think>(?P<answer>.*?)', resp, re.S).group('answer')  # 正则剔除思考内容
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data = [user_id, user_message, llm_message, time]
            # print(data)
            resp_json['user_message'] = user_message
            self.response = resp_json

            # 先写user表
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    insert_query = sql.SQL("""
                                                INSERT INTO users (id, username, pw)  
                                                VALUES (%s, %s, %s)ON CONFLICT (id) DO NOTHING;
                                            """)               # 需要修改是否插入name和pw字段！！
                    cur.execute(insert_query, (user_id, '', ''))
                    conn.commit()
            # 再写消息
            with psycopg2.connect(**conn_params) as conn:
                with conn.cursor() as cur:
                    insert_query = sql.SQL("""
                                                INSERT INTO message_history (user_id, user_message, llm_message, time)
                                                VALUES (%s, %s, %s, %s)
                                            """)
                    cur.execute(insert_query, data)
                    conn.commit()
        return resp


if __name__ == '__main__':
    llm = LLM()
    resp = llm.input('你是谁', need2store=True)
    # print(resp)
    # print(llm.response)