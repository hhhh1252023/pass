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


    def test_api_01_abort_request(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "rid": '123',
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        response = requests.post(f"{DEFAULT_URL_FOR_TEST}/abort_request", json={'rid': '123', 'finished_reason': "test_abort", 'abort_message': "---------AAA---"})
        self.assertEqual(response.status_code, 200)
        print(f'{response.status_code = }') 
        print(response.json())

    """
    def test_api_02_pause_generation(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "rid": '123'
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        response = requests.post(f"{DEFAULT_URL_FOR_TEST}/pause_generation", json={'rid': '123', 'http_worker_ipc': "abc", 'mode': "in_place"})
        #self.assertEqual(response.status_code, 200)
        print(f'{response.status_code = }') 
        print(response.json())
        #self.assertEqual(response.json()['rid'], None)
        #self.assertEqual(response.json()['http_worker_ipc'], None)
        #self.assertEqual(response.json()['mode'], "abort")

    def test_api_03_continue_generation(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "rid": '123'
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/continue_generation", json={'rid': '123', 'http_worker_ipc': "abc"})
        self.assertEqual(response.status_code, 200)
        print(response.json())
        #self.assertEqual(response.json()['rid'], None)
        #self.assertEqual(response.json()['http_worker_ipc'], None)
    """

if __name__ == "__main__":

    unittest.main()
