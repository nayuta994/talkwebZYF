import re
import requests
import pandas as pd
import psycopg2
from psycopg2 import sql

from docs_reader2split import split_tool


def embedding(text_list: list):
    headers = {
        'Authorization': 'Bearer gpustack_0bcd33ceb0c8f708_e5309686ce88e01dbae423bb086f180f',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "bge-m3",
        "input": text_list
    }
    resp = requests.post('http://192.168.143.117:9212/v1-openai/embeddings', headers=headers, json=data).json()
    # print(resp)

    embedding_list = []
    for data in resp['data']:
        embedding_list.append(data['embedding'])
    # print(type(embedding_list[0][0]))  # float
    return zip(text_list, embedding_list)


def save_2_csv(data: tuple):
    content_list = []
    embedding_list = []
    for c, e in data:
        content_list.append(c)
        embedding_list.append(e)

    new_data_dict = {
        'content': content_list,
        'embedding': embedding_list
    }
    new_df = pd.DataFrame(new_data_dict)

    try:
        # 尝试读取现有的CSV文件
        existing_df = pd.read_csv('embedding.csv')
        # 将新数据追加到现有数据中
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    except FileNotFoundError:
        # 如果文件不存在，直接使用新数据
        combined_df = new_df

    # 保存合并后的数据到CSV文件
    combined_df.to_csv('embedding.csv', index=False)


def save_2_pgvector(data):
    # 数据库连接参数
    conn_params = {
        "host": "192.168.143.117",
        "port": "9002",
        "database": "pgvector",
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
    print("嵌入数据已成功保存到 pgvector 数据库。")


if __name__ == '__main__':
    # 读取文本
    text_list = []
    with open('content.txt', mode='r', encoding='utf-8') as f:
        content = f.read()
        paragraphs = re.split(r'\n+', content)  # 使用正则表达式分割一个或多个换行符
        for paragraph in paragraphs:
            text_list.append(paragraph.strip())
    text = ''.join(text_list).strip()
    splits = split_tool([text],6300,200)
    print(splits)
    # text_list = ["What is Task Decomposition?"]
    # 词嵌入、保存
    result = embedding([split.page_content for split in splits])  # 维度是1024
    # print(result)
    # save_2_csv(result)
    save_2_pgvector(list(result))
