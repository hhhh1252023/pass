import unittest
import requests
import time
import re
import os
import sys
from types import SimpleNamespace
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_URL_FOR_TEST = "http://127.0.0.1:8234"

# 定义日志关键字匹配模式
DYNAMIC_BATCH_LOG_PATTERN = r"AsyncDynamicbatchTokenizer: Processing dynamic batch of size (\d+)"

class TestQwenPPTieWeightsAccuracy(CustomTestCase):
    # 基础配置
    accuracy = 0.38
    base_url = DEFAULT_URL_FOR_TEST
    model_name = "/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/Qwen/Qwen3-32B"
    
    # 类级变量，用于存储进程和日志
    process = None
    log_capture = None

    @classmethod
    def _launch_server_with_config(cls, timeout_value):
        """
        通用的服务器启动方法，根据timeout值配置参数
        """
        # 清理之前的进程
        if cls.process:
            kill_process_tree(cls.process.pid)
            cls.process = None
        
        # 构建启动参数
        other_args = [
            "--chunked-prefill-size", "256",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static", "0.8",
            "--tp-size", "4",
            "--base-gpu-id", "4",
            # 动态批处理分词器核心配置
            "--enable-dynamic-batch-tokenizer",
            "--dynamic-batch-tokenizer-batch-size", "4",
            "--dynamic-batch-tokenizer-batch-timeout", str(timeout_value),
            "--log-level", "debug"
        ]
        
        # 启动服务器并捕获日志
        cls.log_capture = StringIO()
        with redirect_stdout(cls.log_capture), redirect_stderr(cls.log_capture):
            cls.process = popen_launch_server(
                cls.model_name,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )
        
        # 等待服务器启动和日志生成
        time.sleep(5)
        
        # 验证服务器是否正常启动
        try:
            response = requests.get(cls.base_url + "/get_server_info", timeout=10)
            self.assertTrue(response.status_code == 200, "服务器启动失败")
        except Exception as e:
            self.fail(f"服务器连接失败: {str(e)}")

    @classmethod
    def tearDownClass(cls):
        """清理进程"""
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _check_accuracy(self):
        """通用精度检查方法"""
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
        self.assertGreaterEqual(
            metrics["accuracy"],
            self.accuracy,
            f'Accuracy of {self.model_name} is {metrics["accuracy"]}, lower than threshold {self.accuracy}',
        )
        
        # 输出服务器信息
        server_info = requests.get(self.base_url + "/get_server_info")
        print(f"Server info response: {server_info.json() if server_info.ok else server_info.text}")

    def _check_dynamic_batch_log(self, expected_batch_size):
        """
        检查日志中是否包含指定的动态批处理关键字
        :param expected_batch_size: 期望的batch size数值
        """
        # 获取日志内容
        log_content = self.log_capture.getvalue()
        
        # 匹配日志模式
        matches = re.findall(DYNAMIC_BATCH_LOG_PATTERN, log_content)
        self.assertTrue(len(matches) > 0, "未找到动态批处理日志关键字")
        
        # 验证batch size是否符合预期
        batch_sizes = [int(m) for m in matches]
        self.assertTrue(
            expected_batch_size in batch_sizes,
            f"日志中未找到batch size {expected_batch_size}，找到的batch sizes: {batch_sizes}"
        )
        print(f"验证通过：日志中找到动态批处理记录，batch size包含 {expected_batch_size}")

    def test_gsm8k_dynamic_batch_timeout_1(self):
        """测试场景1：timeout=1，验证batch size=4和精度"""
        # 启动服务器（timeout=1）
        self._launch_server_with_config(timeout_value=1)
        
        # 检查日志中的batch size
        self._check_dynamic_batch_log(expected_batch_size=4)
        
        # 检查精度
        self._check_accuracy()

    def test_gsm8k_dynamic_batch_timeout_0(self):
        """测试场景2：timeout=0，验证batch size=1和精度"""
        # 启动服务器（timeout=0）
        self._launch_server_with_config(timeout_value=0)
        
        # 检查日志中的batch size
        self._check_dynamic_batch_log(expected_batch_size=1)
        
        # 检查精度
        self._check_accuracy()

if __name__ == "__main__":
    # 设置unittest运行参数，支持详细输出
    unittest.main(verbosity=2)
