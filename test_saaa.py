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

# 封装通用的服务器启动逻辑，接收分词器超时参数
def launch_server_with_tokenizer_timeout(model_name, base_url, tokenizer_timeout, other_args_base):
    """
    启动服务器，核心调整dynamic-batch-tokenizer-batch-timeout参数
    :param tokenizer_timeout: --dynamic-batch-tokenizer-batch-timeout的值
    """
    # 拼接最终的启动参数（替换分词器超时值）
    other_args = other_args_base.copy()
    # 找到并替换dynamic-batch-tokenizer-batch-timeout的值
    if "--dynamic-batch-tokenizer-batch-timeout" in other_args:
        idx = other_args.index("--dynamic-batch-tokenizer-batch-timeout") + 1
        other_args[idx] = str(tokenizer_timeout)
    
    process = popen_launch_server(
        model_name,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  # 保留服务器启动的默认超时
        other_args=other_args,
    )
    return process

# 基础参数（两个测试类共用）
BASE_OTHER_ARGS = [
    "--chunked-prefill-size", "256",
    "--attention-backend", "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static", "0.8",
    "--tp-size", "4",
    "--base-gpu-id", "4",
    "--enable-dynamic-batch-tokenizer",
    "--dynamic-batch-tokenizer-batch-size", "4",
    "--dynamic-batch-tokenizer-batch-timeout", "0",  # 基准值
    "--log-level", "debug"
]
MODEL_NAME = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-32B"

# 测试类1：原有逻辑（--dynamic-batch-tokenizer-batch-timeout=0）
class TestQwenPPTieWeightsAccuracyTokenizerTimeout0(CustomTestCase):
    accuracy = 0.38
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # 启动服务器，分词器超时=0（原有值）
        cls.process = launch_server_with_tokenizer_timeout(
            MODEL_NAME, cls.base_url, tokenizer_timeout=0, other_args_base=BASE_OTHER_ARGS
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_tokenizer_timeout_0(self):
        self._run_gsm8k_test("tokenizer_timeout=0")

# 测试类2：新增逻辑（--dynamic-batch-tokenizer-batch-timeout=1）
class TestQwenPPTieWeightsAccuracyTokenizerTimeout1(CustomTestCase):
    accuracy = 0.38
    
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # 启动服务器，分词器超时=1（你要测试的场景）
        cls.process = launch_server_with_tokenizer_timeout(
            MODEL_NAME, cls.base_url, tokenizer_timeout=1, other_args_base=BASE_OTHER_ARGS
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_tokenizer_timeout_1(self):
        self._run_gsm8k_test("tokenizer_timeout=1")

    # 封装通用测试逻辑，避免重复
    def _run_gsm8k_test(self, scenario):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        
        # 断言精度不低于阈值
        self.assertGreater(
            metrics["accuracy"],
            self.accuracy,
            f'{scenario}场景下，{MODEL_NAME}精度 {metrics["accuracy"]} 低于阈值 {self.accuracy}',
        )
        
        # 调用服务器信息接口，输出相关信息
        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"{scenario} - 服务器信息: {server_info=}")

if __name__ == "__main__":
    unittest.main()
