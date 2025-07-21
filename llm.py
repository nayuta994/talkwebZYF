import psycopg2
import requests
import json

from psycopg2 import sql


def __store_chat_message_to_db(data):
    # 数据库连接参数
    conn_params = {
        "host": "192.168.143.117",
        "port": "9002",
        "database": "history_message",
        "user": "postgres",
        "password": "123456"
    }
    # print(data)

    # 使用 with 语句来管理数据库连接和游标
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            # 准备插入数据的 SQL 语句
            insert_query = sql.SQL("""
                        INSERT INTO embeddings_documents (content, embedding)
                        VALUES (%s, %s)
                    """)
            # 执行 SQL 命令
            for q, e in data:
                cur.execute(insert_query, (q, e))
            conn.commit()


def llm(question, user_id='', prompt='', temperature=0.3, top_p=0.4, stream=False):
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

    # json会返回'id': 'chatcmpl-341c93aa6beb40bfa5c564a830e5d059'
    if user_id == '':
        user_id = resp_json['id']
        user_message = question
        llm_message = resp
        data = [user_id, ]
        __store_chat_message_to_db(data)

    return resp


if __name__ == '__main__':
    print(llm('你好！', ''))
