# -*- coding: utf-8 -*-
import requests
import time
from docs_reader2split import split_tool


def pdf_parse(pdf_path="inputs/肺栓塞中期报告.pdf"):
    """
    旧版minerU解析
    """
    # 定义目标 URL
    url = "http://192.168.143.117:8620/v1/mineru/pdf"

    # 定义请求头
    headers = {
        "accept": "application/json",
    }

    # 定义要上传的文件路径和 MIME 类型
    file_path = pdf_path
    files = {
        "file": (f"{file_path.split('/')[1]}", open(file_path, "rb"))
    }
    # print(files.values())

    # 发送 POST 请求
    start = time.time()
    # 发送 POST 请求
    response = requests.post(url, headers=headers, files=files)
    end = time.time()
    print(end - start)

    # 输出响应内容
    print("状态码:", response.status_code)
    print("响应内容:", response.text)

    return response.json()['result']

    # 关闭文件
    files["file"][1].close()


if __name__ == '__main__':
    text = pdf_parse('inputs/肺栓塞中期报告.pdf')
    for i,s in enumerate(split_tool([text], 200, 5)):
        print(f'No.{i+1}：\n{s.page_content}'.strip() , end='\n-----------------------------------------------------------------------------------\n')