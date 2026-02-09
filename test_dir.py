import unittest
import requests
import os
import glob
from sglang.test.ascend.test_ascend_utils import run_command
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=200, suite="nightly-1-npu-a3", nightly=True)


class TestDownloadDir(CustomTestCase):
    """Testcase：Verify --download-dir and --revision parameters take effect for small model.

       [Test Category] Parameter
       [Test Target] --download-dir, --revision
       """
    # 轻量模型（1.4GB）
    model = "microsoft/Phi-1.5"
    # 100%有效revision哈希（主分支稳定版本）
    revision = "675aa382d814580b22651a30acb1a585d7c25963"
    download_dir = "./phi1.5_weight"

    @classmethod
    def setUpClass(cls):
        # 1. 配置国内镜像（确保下载成功）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # 2. 创建下载目录
        run_command(f"mkdir -p {cls.download_dir}")
        
        # 3. 启动服务（指定有效revision）
        other_args = [
            "--download-dir",
            cls.download_dir,
            "--revision",
            cls.revision,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        # 安全清理
        if hasattr(cls, 'process') and cls.process:
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.download_dir):
            run_command(f"rm -rf {cls.download_dir}")

    def test_download_dir_and_revision(self):
        # 1. 发送推理请求（轻量模型快速响应）
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
            timeout=30
        )
        self.assertEqual(response.status_code, 200, msg="推理请求失败，HTTP状态码非200")
        self.assertIn("Paris", response.text, msg="推理结果错误，未包含'Paris'")

        # 2. 验证--download-dir生效（权重文件存在）
        weight_suffixes = ("*.safetensors", "*.bin", "*.pth")
        weight_files = []
        for suffix in weight_suffixes:
            weight_files.extend(glob.glob(os.path.join(self.download_dir, "**", suffix), recursive=True))
        self.assertGreater(
            len(weight_files),
            0,
            msg=f"--download-dir {self.download_dir} 未找到权重文件，参数未生效"
        )

        # 3. 验证--revision生效（快照目录匹配）
        snapshot_dir = os.path.join(
            self.download_dir,
            f"models--{self.model.replace('/', '--')}",
            "snapshots",
            self.revision
        )
        self.assertTrue(
            os.path.exists(snapshot_dir),
            msg=f"Revision快照目录{snapshot_dir}不存在，--revision参数未生效"
        )
        print(f"✅ 所有参数验证通过：--download-dir和--revision均生效")


if __name__ == "__main__":
    unittest.main()
