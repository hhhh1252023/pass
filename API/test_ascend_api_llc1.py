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

    def test_api_pause_generation(self):
        response = requests.post(f"{DEFAULT_URL_FOR_TEST}/pause_generation",, json={'rid': '123', 'mode': "in_place"})
        #self.assertEqual(response.status_code, 200)
        print(f'{response.status_code = }') 
        print(response.json())
        #self.assertEqual(response.json()['rid'], None)
        #self.assertEqual(response.json()['http_worker_ipc'], None)
        #self.assertEqual(response.json()['mode'], "abort")

    def test_api_continue_generation(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/continue_generation")
        self.assertEqual(response.status_code, 200)
        print(response.json())
        #self.assertEqual(response.json()['rid'], None)
        #self.assertEqual(response.json()['http_worker_ipc'], None)


if __name__ == "__main__":

    unittest.main()
