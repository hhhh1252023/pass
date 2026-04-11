import multiprocessing as mp
import unittest
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# CI 注册
register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

MODEL = "/home/weights/Qwen/Qwen3-VL-30B-A3B-Instruct/"


# 测试视频
p3="/home/l30079981/dataset/test_video.mp4"
VIDEO_JOBS_URL = f"file://{p3}"



def popen_launch_server_wrapper(base_url, model, other_args):
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


class TestSingleVideoDPEncoder(CustomTestCase):
    """测试：单视频请求 + mm-enable-dp-encoder + 完整视频处理配置"""

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        #cls.base_url = DEFAULT_URL_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST + "/v1"

        mm_process_config = '''{
            "image": "",
            "video": {
                "min_pixels": 76800,
                "max_pixels": 921600,
                "resized_height": 448,
                "resized_width": 448,
                "fps": 2,
                "min_frames": 4,
                "max_frames": 64
            },
            "audio": ""
        }'''

        other_args = [
            "--mem-fraction-static", "0.5",
            "--disable-cuda-graph",
            "--attention-backend", "ascend",
            "--device", "npu",
            "--tp-size", "4",
            "--disable-cuda-graph",
            "--base-gpu-id", "8" ,
            "--mm-enable-dp-encoder",
            "--mm-process-config", mm_process_config,
        ]

        cls.process = popen_launch_server_wrapper(
            DEFAULT_URL_FOR_TEST, MODEL, other_args
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_single_video_request(self):
        print("\n===== 发送单个视频推理请求 =====")

        messages = [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": VIDEO_JOBS_URL}},
                {"type": "text", "text": "Describe this video in one sentence."},
            ]
        }]

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "messages": messages,
                "temperature": 0,
                "max_completion_tokens": 1024
            }
        )

        self.assertEqual(response.status_code, 200)
        print(f"请求成功，状态码：{response.status_code}")
        print(f"回复：{response.json()['choices'][0]['message']['content']}\n")


if __name__ == "__main__":
    unittest.main()

