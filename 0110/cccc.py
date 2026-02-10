import os
import unittest
import shutil
from types import SimpleNamespace

# ====================== 核心路径替换 - 改为当前目录 ======================
# 当前目录（绝对路径）
CURRENT_DIR = os.path.abspath(".")
# 模型路径：当前目录下的 Qwen3-30B-A3B 文件夹
QWEN3_30B_A3B_W8A8_WEIGHTS_PATH = os.path.join(CURRENT_DIR, "Qwen3-30B-A3B")
# Crash dump 文件夹：当前目录下的 crash_dump_folder
CRASH_DUMP_FOLDER = os.path.join(CURRENT_DIR, "crash_dump_folder")

from sglang.test.ascend.test_ascend_utils import test_config  # 仅保留必要导入，路径已替换
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestDeepepLowlatencyQwen3(CustomTestCase):
    """Testcase: Verify the accuracy of Qwen3-30B model on MMLU and GSM8K tasks with DeepEP low latency mode on Ascend backend.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend;--deepep-mode
    """
    @classmethod
    def setUpClass(cls):
        # 初始化 crash dump 文件夹（确保存在）
        os.makedirs(CRASH_DUMP_FOLDER, exist_ok=True)
        
        # 模型路径使用当前目录下的路径
        cls.model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        
        # 启动服务：添加 crash-dump-folder 参数，指向当前目录下的文件夹
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
                # 关键：指定 crash dump 路径为当前目录下的文件夹
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

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.check_and_clean_crash_files()

    @classmethod
    def check_and_clean_crash_files(cls):
        if os.path.exists(CRASH_DUMP_FOLDER):
            for item in os.listdir(CRASH_DUMP_FOLDER):
                item_path = os.path.join(CRASH_DUMP_FOLDER, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)



if __name__ == "__main__":
    if os.path.exists(CRASH_DUMP_FOLDER):
        shutil.rmtree(CRASH_DUMP_FOLDER)
    unittest.main()
