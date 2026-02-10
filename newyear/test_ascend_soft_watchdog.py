import io
import os
import unittest
import requests
from contextlib import contextmanager

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# ===================== 聚焦三个核心环境变量的通用管理函数 =====================
# 定义测试中涉及的核心环境变量列表（精准管控）
TEST_RELATED_ENVS = [
    "SGLANG_IS_IN_CI",
    "SGLANG_TEST_STUCK_DETOKENIZER",
    "SGLANG_TEST_STUCK_TOKENIZER"
]

@contextmanager
def temporary_test_envs(ci_mode: bool = None, stuck_detokenizer: int = None, stuck_tokenizer: int = None):
    """
    仅管理测试相关的三个环境变量，自动保存/还原原始值
    :param ci_mode: 是否设置为CI环境（True/False/None：None表示不修改）
    :param stuck_detokenizer: 解词器阻塞时长（int/None：None表示不修改）
    :param stuck_tokenizer: 分词器阻塞时长（int/None：None表示不修改）
    """
    # 步骤1：保存三个变量的原始值（一次性保存，避免遗漏）
    original_values = {
        var: os.environ.get(var) 
        for var in TEST_RELATED_ENVS
    }

    try:
        # 步骤2：按需设置临时值（只修改传入了参数的变量）
        if ci_mode is not None:
            os.environ["SGLANG_IS_IN_CI"] = "True" if ci_mode else "False"
        if stuck_detokenizer is not None:
            os.environ["SGLANG_TEST_STUCK_DETOKENIZER"] = str(stuck_detokenizer)
        if stuck_tokenizer is not None:
            os.environ["SGLANG_TEST_STUCK_TOKENIZER"] = str(stuck_tokenizer)
        
        yield  # 执行with块内的测试代码

    finally:
        # 步骤3：无论成败，还原所有三个变量的原始值（精准还原）
        for var_name, original_val in original_values.items():
            if original_val is None:
                # 原始无该变量 → 删除临时设置的变量
                os.environ.pop(var_name, None)
            else:
                # 原始有值 → 恢复原值
                os.environ[var_name] = original_val

# ===================== 测试用例部分 =====================
class BaseTestSoftWatchdog:
    env_override = None
    expected_message = None

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        # 仅保留--skip-server-warmup，不设置soft-watchdog-timeout
        other_args = [
            "--skip-server-warmup",
        ]

        # 应用环境变量覆盖（模拟组件阻塞）
        with cls.env_override():
            cls.process = popen_launch_server(
                "Qwen/Qwen3-0.6B",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_watchdog_triggers(self):
        print("Start call /generate API", flush=True)
        # 直接调用API（无try-except）
        requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello, please repeat this sentence for 1000 times.",
                "sampling_params": {"max_new_tokens": 100, "temperature": 0},
            },
            timeout=30,
        )
        print("End call /generate API", flush=True)

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        
        # 核心验证逻辑
        if os.environ.get("SGLANG_IS_IN_CI") == "True":
            self.assertIn(
                self.expected_message, 
                combined_output,
                f"CI环境下未找到预期的超时日志: {self.expected_message}"
            )
        else:
            self.assertNotIn(
                self.expected_message, 
                combined_output,
                f"非CI环境下意外出现超时日志: {self.expected_message}"
            )

class TestSoftWatchdogDetokenizer(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_DETOKENIZER.override(350)
    expected_message = "DetokenizerManager watchdog timeout"

class TestSoftWatchdogTokenizer(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_TOKENIZER.override(350)
    expected_message = "TokenizerManager watchdog timeout"

class TestSoftWatchdogSchedulerInit(BaseTestSoftWatchdog, CustomTestCase):
    env_override = lambda: envs.SGLANG_TEST_STUCK_DETOKENIZER.override(350)
    expected_message = "DetokenizerManager watchdog timeout"

# ===================== 测试执行函数（使用通用环境变量管理） =====================
def run_test_by_env(ci_mode: bool, test_case_cls):
    """
    根据环境配置运行指定测试用例（使用通用函数管理环境变量）
    :param ci_mode: 是否为CI环境
    :param test_case_cls: 测试用例类
    """
    # 使用封装的上下文管理器，仅管理三个测试相关变量
    with temporary_test_envs(ci_mode=ci_mode):
        # 运行测试
        suite = unittest.TestLoader().loadTestsFromTestCase(test_case_cls)
        unittest.TextTestRunner(verbosity=2).run(suite)

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 步骤1: CI环境 - 解词器阻塞测试
    print("=== Step 1: CI环境 - 解词器阻塞测试 ===")
    run_test_by_env(ci_mode=True, test_case_cls=TestSoftWatchdogDetokenizer)
    
    # 步骤2: CI环境 - 分词器阻塞测试
    print("\n=== Step 2: CI环境 - 分词器阻塞测试 ===")
    run_test_by_env(ci_mode=True, test_case_cls=TestSoftWatchdogTokenizer)
    
    # 步骤3: 非CI环境 - 解词器阻塞测试
    print("\n=== Step 3: 非CI环境 - 解词器阻塞测试 ===")
    run_test_by_env(ci_mode=False, test_case_cls=TestSoftWatchdogDetokenizer)
