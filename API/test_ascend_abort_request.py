import json
import threading
import requests


responses = []

def send_requests(url, **kwargs):
    data = json.dumps(kwargs)
    response = requests.post('http://127.0.0.1:12345' + url, json=data)
    responses.append(response)


thread1 = threading.Thread(target=send_requests, args=('/generate',), kwargs={'rid': '10086', 'text': 'who are you?', 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1024}})
thread2 = threading.Thread(target=send_requests, args=('/abort_request',), kwargs={'rid': "10086"})

thread1.start()
thread2.start()
thread1.join()
thread2.join()
