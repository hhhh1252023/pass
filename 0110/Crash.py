import os
import unittest
import shutil
import time

# ====================== 核心路径配置 ======================
CURRENT_DIR = os.path.abspath(".")
QWEN3_30B_A3B_W8A8_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "Qwen3-30B-A3B")
CRASH_DUMP_FOLDER = os.path.join(CURRENT_DIR, "crash_dump_folder")

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestDeepepLowlatencyQwen3(CustomTestCase):
    """Testcase: Check if crash dump log files are generated for Qwen3-30B with DeepEP low latency mode."""
    @classmethod
    def setUpClass(cls):
        # 清空残留的 crash dump 文件夹
        if os.path.exists(CRASH_DUMP_FOLDER):
            shutil.rmtree(CRASH_DUMP_FOLDER)
        # 重新创建空的 crash dump 文件夹
        os.makedirs(CRASH_DUMP_FOLDER, exist_ok=True)
        
        # 启动服务（核心：触发 crash 日志生成）
        cls.model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--quantization",
                "modelslim",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "low_latency",
                "--disable-cuda-graph",
                "--chunked-prefill-size",
                "1024",
                "--crash-dump-folder", CRASH_DUMP_FOLDER,
            ],
            env={
                "HCCL_BUFFSIZE": "1536",
                "SGLANG_SET_CPU_AFFINITY": "1",
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
                "SGLANG_DEEPEP_BF16_DISPATCH": "1",
                "ENABLE_ASCEND_MOE_NZ": "1",
                "SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS": "1",
                **os.environ,
            },
        )
        # 等待日志生成（根据实际情况调整等待时间，单位：秒）
        time.sleep(30)

    @classmethod
    def tearDownClass(cls):
        # 终止服务进程
        kill_process_tree(cls.process.pid)
        # 清理 crash dump 文件
        if os.path.exists(CRASH_DUMP_FOLDER):
            for item in os.listdir(CRASH_DUMP_FOLDER):
                item_path = os.path.join(CRASH_DUMP_FOLDER, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

    # 核心测试方法：仅检查 crash dump 日志文件是否存在
    def test_crash_log_generated(self):
        # 检查 crash dump 文件夹下是否有文件/子文件夹
        has_crash_files = False
        if os.path.exists(CRASH_DUMP_FOLDER):
            # 遍历文件夹，检测是否有 crash dump 相关文件
            for root, dirs, files in os.walk(CRASH_DUMP_FOLDER):
                if dirs or files:  # 有子文件夹或文件即判定为日志生成
                    has_crash_files = True
                    break
        
        # 断言：验证日志文件已生成（核心检查点）
        self.assertTrue(
            has_crash_files,
            f"Crash dump files are not generated in {CRASH_DUMP_FOLDER}"
        )


if __name__ == "__main__":
    unittest.main()
