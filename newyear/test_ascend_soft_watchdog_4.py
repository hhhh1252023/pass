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

# 通用环境变量管理函数
TEST_RELATED_ENVS = [
    "SGLANG_IS_IN_CI",
    "SGLANG_TEST_STUCK_DETOKENIZER"
]

@contextmanager
def temporary_test_envs(ci_mode: bool = None, stuck_detokenizer: int = None):
    """仅管理测试相关的环境变量，自动保存/还原"""
    original_values = {var: os.environ.get(var) for var in TEST_RELATED_ENVS}
    try:
        if ci_mode is not None:
            os.environ["SGLANG_IS_IN_CI"] = "True" if ci_mode else "False"
        if stuck_detokenizer is not None:
            os.environ["SGLANG_TEST_STUCK_DETOKENIZER"] = str(stuck_detokenizer)
        yield
    finally:
        for var_name, original_val in original_values.items():
            if original_val is None:
                os.environ.pop(var_name, None)
            else:
                os.environ[var_name] = original_val

# 基础测试类（聚焦DetokenizerManager）
class BaseTestDetokenizerWatchdog:
    ci_mode = None          # 是否为CI环境
    set_soft_watchdog = None# 是否设置soft-watchdog-timeout
    soft_watchdog_value = 10# 设置时的默认值（子类可覆盖）
    stuck_seconds = 350     # 解词器阻塞时长（子类可覆盖）
    expected_log = None     # 预期日志
    expected_assert_msg = "stuck tester can be enabled only if soft watchdog is enabled"
    error_found_in_log = False  # 日志中是否找到预期错误

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = None
        cls.launch_success = False

        # 构建启动参数
        other_args = ["--skip-server-warmup"]
        if cls.set_soft_watchdog:
            other_args.extend(["--soft-watchdog-timeout", str(cls.soft_watchdog_value)])

        # 场景4单独设置20秒超时（保证日志完整），其他场景用默认超时
        timeout = 20 if (cls.ci_mode is False and cls.set_soft_watchdog is False) else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        
        try:
            # 模拟解词器阻塞
            with envs.SGLANG_TEST_STUCK_DETOKENIZER.override(cls.stuck_seconds):
                cls.process = popen_launch_server(
                    "Qwen/Qwen3-0.6B",
                    DEFAULT_URL_FOR_TEST,
                    timeout=timeout,
                    other_args=other_args,
                    return_stdout_stderr=(cls.stdout, cls.stderr),
                )
            cls.launch_success = True
        except TimeoutError:
            # 场景4预期超时，检查日志中的断言错误
            cls.launch_success = False
            combined_log = cls.stdout.getvalue() + cls.stderr.getvalue()
            if cls.expected_assert_msg in combined_log:
                cls.error_found_in_log = True
        finally:
            # 兜底清理进程
            if cls.process:
                kill_process_tree(cls.process.pid)

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        if cls.stdout:
            cls.stdout.close()
        if cls.stderr:
            cls.stderr.close()

    def test_detokenizer_watchdog(self):
        # 场景4：非CI+不设置软看门狗 → 验证日志中的AssertionError
        if self.ci_mode is False and self.set_soft_watchdog is False:
            self.assertTrue(
                self.error_found_in_log,
                f"未找到预期错误: {self.expected_assert_msg}"
            )
            return

        # 场景1-3：启动成功 → 调用API并验证超时日志
        self.assertTrue(self.launch_success, "服务启动失败")
        
        requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello, please repeat this sentence for 1000 times.",
                "sampling_params": {"max_new_tokens": 100, "temperature": 0},
            },
            timeout=40,
        )

        # 验证预期日志
        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(
            self.expected_log,
            combined_output,
            f"未找到预期日志: {self.expected_log}"
        )

# 四个场景的测试子类
class TestCIWithoutSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = True
    set_soft_watchdog = False
    stuck_seconds = 350
    expected_log = "DetokenizerManager watchdog timeout"

class TestCIWithSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = True
    set_soft_watchdog = True
    soft_watchdog_value = 20
    stuck_seconds = 30
    expected_log = "DetokenizerManager watchdog timeout"

class TestNonCIWithSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = False
    set_soft_watchdog = True
    soft_watchdog_value = 20
    stuck_seconds = 30
    expected_log = "DetokenizerManager watchdog timeout"

class TestNonCIWithoutSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = False
    set_soft_watchdog = False

# 测试执行函数
def run_test_scenario(test_case_cls):
    """运行单个测试场景，自动管理环境变量"""
    with temporary_test_envs(ci_mode=test_case_cls.ci_mode):
        suite = unittest.TestLoader().loadTestsFromTestCase(test_case_cls)
        unittest.TextTestRunner(verbosity=2).run(suite)

# 主函数（执行四个场景）
if __name__ == "__main__":
    print("=== 场景1: CI环境 - 不设置soft-watchdog ===")
    run_test_scenario(TestCIWithoutSoftWatchdog)

    print("\n=== 场景2: CI环境 - 设置soft-watchdog(20秒)，阻塞30秒 ===")
    run_test_scenario(TestCIWithSoftWatchdog)

    print("\n=== 场景3: 非CI环境 - 设置soft-watchdog(20秒)，阻塞30秒 ===")
    run_test_scenario(TestNonCIWithSoftWatchdog)

    print("\n=== 场景4: 非CI环境 - 不设置soft-watchdog（验证断言错误） ===")
    run_test_scenario(TestNonCIWithoutSoftWatchdog)
