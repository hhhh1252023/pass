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
                "--disable-cuda-graph",
                "--tp-size",
                2,
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

    def test_api_clear_hicache_storage_backend(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/clear_hicache_storage_backend")
        self.assertEqual(response.status_code, 200)
        print(response.json())
        self.assertEqual(response.json(), "Hierarchical cache storage backend cleared.")

    def test_api_set_internal_state(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/set_internal_state")
        self.assertEqual(response.status_code, 200)
        print(response.json())
        self.assertEqual(response.json(), "Hierarchical cache storage backend cleared.")

    def test_api_configure_logging(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/configure_logging")
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/configure_logging",
            json={
                "log_requests": "True",
            },
        )
        self.assertEqual(response.status_code, 200)
        print(response.json())
        self.assertEqual(response.json()['log_requests'], self.model)
        self.assertEqual(response.json()['log_requests_level'][0]['object'], "model")
        self.assertEqual(response.json()['dump_requests_folder'][0]['owned_by'], "sglang")
        self.assertEqual(response.json()['dump_requests_threshold'][0]['root'], self.model)
        self.assertEqual(response.json()['crash_dump_folder'][0]['max_model_len'], 131072)

if __name__ == "__main__":

    unittest.main()



