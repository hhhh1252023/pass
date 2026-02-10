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

# ===================== 通用环境变量管理函数 =====================
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

# ===================== 基础测试类（聚焦DetokenizerManager） =====================
class BaseTestDetokenizerWatchdog:
    ci_mode = None          # 是否为CI环境
    set_soft_watchdog = None# 是否设置soft-watchdog-timeout
    soft_watchdog_value = 10# 设置时的默认值（子类可覆盖）
    stuck_seconds = 350     # 解词器阻塞时长（子类可覆盖）
    expected_log = None     # 预期日志/报错
    expected_error = None   # 预期异常类型

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = None
        cls.launch_success = False
        cls.assertion_error_caught = False  # 标记是否捕获到预期的AssertionError

        # 构建启动参数（是否设置soft-watchdog-timeout）
        other_args = ["--skip-server-warmup"]
        if cls.set_soft_watchdog:
            other_args.extend(["--soft-watchdog-timeout", str(cls.soft_watchdog_value)])

        try:
            # 模拟解词器阻塞
            with envs.SGLANG_TEST_STUCK_DETOKENIZER.override(cls.stuck_seconds):
                # 场景4缩短超时时间，快速终止
                timeout = 5 if (cls.ci_mode is False and cls.set_soft_watchdog is False) else DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
                cls.process = popen_launch_server(
                    "Qwen/Qwen3-0.6B",
                    DEFAULT_URL_FOR_TEST,
                    timeout=timeout,
                    other_args=other_args,
                    return_stdout_stderr=(cls.stdout, cls.stderr),
                )
            cls.launch_success = True
        except AssertionError as e:
            # 捕获到预期的断言错误：记录+标记+打印日志
            cls.expected_error = e
            cls.assertion_error_caught = True
            cls.launch_success = False
            print(f"\n【场景4】捕获到预期的AssertionError: {str(e)}")
            # 主动清理进程（即使启动失败，也可能有残留子进程）
            if cls.process:
                kill_process_tree(cls.process.pid)
                print(f"【场景4】已清理残留进程（PID: {cls.process.pid}）")
        except TimeoutError:
            # 仅当未捕获到AssertionError时，才抛出超时错误
            if not cls.assertion_error_caught:
                raise
            cls.launch_success = False
            print(f"\n【场景4】服务启动超时（已捕获AssertionError，忽略超时）")

    @classmethod
    def tearDownClass(cls):
        # 最终兜底清理：确保所有进程都被终止
        if cls.process:
            kill_process_tree(cls.process.pid)
            print(f"【场景{cls.__name__}】tearDown清理进程（PID: {cls.process.pid}）")
        if cls.stdout:
            cls.stdout.close()
        if cls.stderr:
            cls.stderr.close()

    def test_detokenizer_watchdog(self):
        # 场景4：非CI+不设置软看门狗 → 验证启动时的AssertionError
        if self.assertion_error_caught:
            # 验证错误信息是否匹配预期
            self.assertIn(
                "stuck tester can be enabled only if soft watchdog is enabled",
                str(self.expected_error),
                "非CI不设置软看门狗未触发预期的AssertionError"
            )
            # 打印完整日志，便于排查
            stderr_output = self.stderr.getvalue()
            stdout_output = self.stdout.getvalue()
            if stderr_output:
                print(f"\n【场景4】STDERR日志: {stderr_output}")
            if stdout_output:
                print(f"\n【场景4】STDOUT日志: {stdout_output}")
            print("【场景4】测试通过：捕获到预期的AssertionError")
            return

        # 场景1-3：启动成功 → 调用API并验证超时日志
        if not self.launch_success:
            self.fail("服务启动失败（未捕获到预期的AssertionError）")
            return

        print("Start call /generate API", flush=True)
        requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello, please repeat this sentence for 1000 times.",
                "sampling_params": {"max_new_tokens": 100, "temperature": 0},
            },
            timeout=40,  # 延长超时时间，确保阻塞30秒能被捕获
        )
        print("End call /generate API", flush=True)

        # 合并输出并验证预期日志
        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(
            self.expected_log,
            combined_output,
            f"未找到预期日志: {self.expected_log}"
        )
        print(f"【场景{self.__class__.__name__}】测试通过：找到预期日志 {self.expected_log}")

# ===================== 四个场景的测试子类 =====================
# 场景1：CI环境 + 不设置soft-watchdog（默认300秒）→ 阻塞350秒
class TestCIWithoutSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = True
    set_soft_watchdog = False
    stuck_seconds = 350  # 保持原有值（超过CI默认300秒）
    expected_log = "DetokenizerManager watchdog timeout"

# 场景2：CI环境 + 设置soft-watchdog（20秒）→ 阻塞30秒（触发超时）
class TestCIWithSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = True
    set_soft_watchdog = True
    soft_watchdog_value = 20  # 软看门狗设为20秒
    stuck_seconds = 30        # 阻塞时长设为30秒（超过20秒触发超时）
    expected_log = "DetokenizerManager watchdog timeout"

# 场景3：非CI环境 + 设置soft-watchdog（20秒）→ 阻塞30秒（触发超时）
class TestNonCIWithSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = False
    set_soft_watchdog = True
    soft_watchdog_value = 20  # 软看门狗设为20秒
    stuck_seconds = 30        # 阻塞时长设为30秒（超过20秒触发超时）
    expected_log = "DetokenizerManager watchdog timeout"

# 场景4：非CI环境 + 不设置soft-watchdog（触发AssertionError）
class TestNonCIWithoutSoftWatchdog(BaseTestDetokenizerWatchdog, CustomTestCase):
    ci_mode = False
    set_soft_watchdog = False

# ===================== 测试执行函数 =====================
def run_test_scenario(test_case_cls):
    """运行单个测试场景，自动管理环境变量"""
    with temporary_test_envs(ci_mode=test_case_cls.ci_mode):
        suite = unittest.TestLoader().loadTestsFromTestCase(test_case_cls)
        unittest.TextTestRunner(verbosity=2).run(suite)

# ===================== 主函数（执行四个场景） =====================
if __name__ == "__main__":
    # 场景1：CI + 不设置soft-watchdog
    print("=== 场景1: CI环境 - 不设置soft-watchdog ===")
    run_test_scenario(TestCIWithoutSoftWatchdog)

    # 场景2：CI + 设置soft-watchdog(20秒) → 阻塞30秒
    print("\n=== 场景2: CI环境 - 设置soft-watchdog(20秒)，阻塞30秒 ===")
    run_test_scenario(TestCIWithSoftWatchdog)

    # 场景3：非CI + 设置soft-watchdog(20秒) → 阻塞30秒
    print("\n=== 场景3: 非CI环境 - 设置soft-watchdog(20秒)，阻塞30秒 ===")
    run_test_scenario(TestNonCIWithSoftWatchdog)

    # 场景4：非CI + 不设置soft-watchdog（验证AssertionError）
    print("\n=== 场景4: 非CI环境 - 不设置soft-watchdog（验证断言错误） ===")
    run_test_scenario(TestNonCIWithoutSoftWatchdog)
