import psycopg2
import requests
import json
from psycopg2 import sql
from requests import RequestException


def __store_chat_message_to_db(data):
    conn_params = {
        "host": "192.168.143.117",
        "port": "9002",
        "database": "history_message",
        "user": "postgres",
        "password": "123456"
    }

    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            insert_query = sql.SQL("""
                INSERT INTO embeddings_documents (content, embedding)
                VALUES (%s, %s)
            """)
            for q, e in data:
                cur.execute(insert_query, (q, e))
            conn.commit()


def stream_response(response):
    # 流式处理
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        # response.iter_content返回一个生成器，每次 yield 一个字节块（bytes 类型，除非 decode_unicode=True）
        # decode_unicode=True-->会自动将字节流解码为 Unicode 字符串
        if chunk:  # 过滤掉 keep-alive 的空 chunk------type:str
            try:
                # print('chunk_str:'+str(chunk))
                # chunk_str:data: {"id":"chatcmpl-6f3e81b4cde246e98214effdf474737a",
                # "object":"chat.completion.chunk","created":1753060607,"model":"qwen3-14b-fp8",
                # "choices":[{"index":0,"delta":{"content":"中文"},"logprobs":null,"finish_reason":null}]}
                # 假设服务器返回的是单独的 JSON 片段
                chunk_json = json.loads(chunk[5:])  # 解析为json方便提取内容
                # print(f'chunk_json:{chunk_json}')
                if "choices" in chunk_json and chunk_json["choices"]:
                    # print('yes')
                    delta = chunk_json["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    # print('content:'+content)
                    if content:
                        yield content  # 使用yield返回生成器
                        # full_response += content
                        # print(content, end='', flush=True)  # 实时输出
            except json.JSONDecodeError:
                continue  # 忽略无法解析的片段


def llm(question, user_id='', prompt='', temperature=0.3, top_p=0.4, stream=False):
    url = 'http://192.168.143.117:9212/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer gpustack_0bcd33ceb0c8f708_e5309686ce88e01dbae423bb086f180f'
    }
    data = {
        "model": "qwen3-14b-fp8",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(data),
            stream=stream
        )
        response.raise_for_status()  # 检查 HTTP 错误
        if stream:
            # 流式处理
            return stream_response(response)  # 返回生成器
        else:
            # 非流式处理
            resp_json = response.json()
            return resp_json['choices'][0]['message']['content']

    except RequestException as e:
        print(f"请求失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"解析响应失败: {e}")
        return None


if __name__ == '__main__':
    print(llm('你好！', '', stream=False))

    for i in llm('你是谁?', '', stream=True):
        print(i, end='')