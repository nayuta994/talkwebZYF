import requests
import json


def llm(question, prompt='', temperature=0.3, top_p=0.4, stream=False):
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
                'content': prompt},
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream
    }
    resp = requests.post(url, headers=headers, data=json.dumps(data), stream=stream).json()['choices'][0]['message'][
        'content']
    return resp


if __name__ == '__main__':
    print(llm('你好！', ''))
