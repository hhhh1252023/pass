import unittest
import requests  # 补充缺失的requests模块导入
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_URL_FOR_TEST = "http://127.0.0.1:8234"

# 封装通用的初始化逻辑，避免代码重复
def launch_server_with_timeout(model_name, base_url, timeout, other_args):
    """通用的服务器启动函数，接收不同的timeout参数"""
    process = popen_launch_server(
        model_name,
        base_url,
        timeout=timeout,
        other_args=other_args,
    )
    return process

# 测试类1：使用默认timeout（原有逻辑）
class TestQwenPPTieWeightsAccuracyDefaultTimeout(CustomTestCase):
    # 保持精度阈值不变
    accuracy = 0.38
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model_name = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-32B"
        cls.other_args = [
            "--chunked-prefill-size", "256",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static", "0.8",
            "--tp-size", "4",
            "--base-gpu-id", "4",
            "--enable-dynamic-batch-tokenizer",
            "--dynamic-batch-tokenizer-batch-size", "4",
            "--dynamic-batch-tokenizer-batch-timeout", "0",
            "--log-level", "debug"
        ]
        # 使用默认timeout
        cls.process = launch_server_with_timeout(
            cls.model_name, cls.base_url, 
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, cls.other_args
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_default_timeout(self):
        args = SimpleNamespace(
            num_shots=5, data_path=None, num_questions=200,
            max_new_tokens=512, parallel=128,
            host="http://127.0.0.1", port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        self.assertGreater(
            metrics["accuracy"], self.accuracy,
            f'默认timeout下精度 {metrics["accuracy"]} 低于阈值 {self.accuracy}'
        )
        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"默认timeout服务器信息: {server_info=}")

# 测试类2：使用timeout=1（新增逻辑）
class TestQwenPPTieWeightsAccuracyTimeout1(CustomTestCase):
    # 保持精度阈值不变
    accuracy = 0.38
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.model_name = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-32B"
        cls.other_args = [
            "--chunked-prefill-size", "256",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static", "0.8",
            "--tp-size", "4",
            "--base-gpu-id", "4",
            "--enable-dynamic-batch-tokenizer",
            "--dynamic-batch-tokenizer-batch-size", "4",
            "--dynamic-batch-tokenizer-batch-timeout", "0",
            "--log-level", "debug"
        ]
        # 使用timeout=1
        cls.process = launch_server_with_timeout(
            cls.model_name, cls.base_url, 
            1, cls.other_args  # 核心修改：timeout=1
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_timeout_1(self):
        args = SimpleNamespace(
            num_shots=5, data_path=None, num_questions=200,
            max_new_tokens=512, parallel=128,
            host="http://127.0.0.1", port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        self.assertGreater(
            metrics["accuracy"], self.accuracy,
            f'timeout=1下精度 {metrics["accuracy"]} 低于阈值 {self.accuracy}'
        )
        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"timeout=1服务器信息: {server_info=}")

if __name__ == "__main__":
    # 运行所有测试类
    unittest.main()
