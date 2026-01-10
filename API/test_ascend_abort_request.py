import json
import threading
import requests
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

responses = []
def send_requests(url, **kwargs):
    data = json.dumps(kwargs)
    response = requests.post('http://127.0.0.1:12345' + url, json=data)
    responses.append(response)

class TestAscendApi(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
        other_args = (
            [
                "--attention-backend",
                "ascend",
            ]
        )
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_api_generate_from_file(self):
        with open('/home/test_embeds_qwen3.json', 'r') as f:
            file = {'file': f}  
            response = requests.post(f"{DEFAULT_URL_FOR_TEST}/generate_from_file", files=file)  
            print(res.text)
        self.assertEqual(response.status_code, 200)
        print(response.json())

    def test_api_abort_request(self):
        thread1 = threading.Thread(target=send_requests, args=('/generate',), kwargs={'rid': '10086', 'text': 'who are you?', 'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1024}})
        thread2 = threading.Thread(target=send_requests, args=('/abort_request',), kwargs={'rid': "10086"})
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        #self.assertEqual(response.status_code, 200)
        print(f'{response.status_code = }') 
        print(response.json())


if __name__ == "__main__":

    unittest.main()
