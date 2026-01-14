import unittest
import base64 
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAscendApi(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "/root/.cache/modelscope/hub/models/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
        other_args = (
            [
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--tp-size",
                2,
                "--is-embedding",
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
        
    def test_api_encode_01(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "23",
                "text": "what is the capital of France",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200,
                    "top_p": 1
                },
                
            },
        )
        print(response.json().keys())
        print(response.json()['embedding'])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['meta_info']['id'], "23")
        #self.assertEqual(response.json()['sampling_params']['temperature'], 0)
    def test_api_encode_02(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": ["8", "88", "888"],
                "text": [  
                    "what is the capital of France",
                    "what is the capital of China",
                    "how to learn Python well"
                ],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200,
                },
                
            },
        )
        print(response.json())
        self.assertEqual(response.status_code, 200)
        #self.assertEqual(response.json()['meta_info']['id'][0], "8")

    def test_api_encode_03(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "3",
                "input_ids": [101, 7592, 2088, 102],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200    
                },
                
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['meta_info']['id'], "3")
        #self.assertEqual(response.json()['sampling_params']['temperature'], 0)
        
    def test_api_encode_04(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "4",
                "text": "show me the words",
                "image_data": "https://miaobi-lite.bj.bcebos.com/miaobi/5mao/b%27b2Ny6K%2BG5Yir5Luj56CBXzE3MzQ2MzcyNjAuMzgxNDk5NQ%3D%3D%27/0.png",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200    
                },
                
            },
        )
        #print(response.json())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['meta_info']['id'], "4")

    def test_api_encode_05(self):
        image_file_path = "/data/d00662834/0104_dev/sglang/examples/assets/example_image.png"  # 替换为你的本地图片绝对路径
        with open(image_file_path, "rb") as f:
            image_binary = f.read()

        image_base64 = base64.b64encode(image_binary).decode("utf-8")
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "44",
                "text": "describe  me the picture",
                "image_data": image_base64,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200    
                },
                
            },
        )
        #print(response.json())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['meta_info']['id'], "44")

    def test_api_encode_06(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/encode",
            json={
                "rid": "48",
                "text": "describe me the picture",
                # 文件路径格式：传入服务端可访问的绝对路径
                "image_data": "/data/d00662834/0104_dev/sglang/examples/assets/example_image.png",  # 替换为服务端上的图片绝对路径
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 200    
                }
            },
        )
        #print(response.json())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['meta_info']['id'], "48")


if __name__ == "__main__":

    unittest.main()
