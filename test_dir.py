import unittest
import requests
import os
import glob
import time
from sglang.test.ascend.test_ascend_utils import run_command
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)


class TestDownloadDir(CustomTestCase):
    """Testcase：Verify --download-dir and specific revision take effect (single inference request).

       [Test Category] Parameter
       [Test Target] --download-dir, --revision (specific commit hash)
       """
    # 注意：模型名是phi-1_5（下划线），与网页路径一致
    model = "microsoft/phi-1_5"
    # 你验证过的有效commit hash（精准版本）
    revision = "675aa382d814580b22651a30acb1a585d7c25963"
    download_dir = "./phi1.5_weight"
    DOWNLOAD_TIMEOUT = 300  # 5分钟下载超时
    WEIGHT_FILE_NAME = "model.safetensors"  # Phi-1.5固定权重文件名

    @classmethod
    def setUpClass(cls):
        # 1. 配置国内镜像（确保精准版本能下载）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # 2. 创建下载目录
        run_command(f"mkdir -p {cls.download_dir}")
        
        # 3. 启动服务（指定精准的revision hash）
        other_args = [
            "--download-dir", cls.download_dir,
            "--revision", cls.revision,  # 用你验证过的有效hash
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--trust-remote-code",
        ]
        cls.process = popen_launch_server(
            cls.model, DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        # 安全清理进程和目录
        if hasattr(cls, 'process') and cls.process:
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.download_dir):
            run_command(f"rm -rf {cls.download_dir}")

    def wait_for_weight_download(self):
        """等待权重下载完成（核心逻辑）"""
        start_time = time.time()
        weight_found = False
        
        while time.time() - start_time < self.DOWNLOAD_TIMEOUT:
            weight_files = glob.glob(
                os.path.join(self.download_dir, "**", self.WEIGHT_FILE_NAME),
                recursive=True
            )
            if weight_files:
                weight_found = True
                print(f"✅ 权重文件下载完成：{weight_files[0]}")
                break
            time.sleep(5)
            elapsed = int(time.time() - start_time)
            print(f"等待权重下载中...已耗时{elapsed}秒（超时{self.DOWNLOAD_TIMEOUT}秒）")
        
        if not weight_found:
            raise TimeoutError(f"❌ 权重下载超时，未找到 {self.WEIGHT_FILE_NAME}")

    def test_download_dir_and_revision(self):
        # 第一步：等待权重下载完成（确保文件存在）
        self.wait_for_weight_download()

        # 第二步：单次推理请求（无重试，符合你的要求）
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=30
        )
        self.assertEqual(response.status_code, 200, msg="推理请求失败，HTTP状态码非200")
        self.assertIn("Paris", response.text, msg="推理结果错误，未包含'Paris'")

        # 第三步：验证--download-dir参数生效
        weight_suffixes = ("*.safetensors", "*.bin", "*.pth")
        weight_files = []
        for suffix in weight_suffixes:
            weight_files.extend(glob.glob(os.path.join(self.download_dir, "**", suffix), recursive=True))
        
        self.assertGreater(
            len(weight_files),
            0,
            msg=f"--download-dir {self.download_dir} 未找到权重文件，参数未生效"
        )
        print(f"✅ 找到{len(weight_files)}个权重文件，--download-dir参数生效")

        # 第四步：验证精准revision参数生效（检查快照目录是否匹配hash）
        snapshot_dir = os.path.join(
            self.download_dir,
            f"models--{self.model.replace('/', '--')}",
            "snapshots",
            self.revision  # 精准匹配你指定的hash目录
        )
        self.assertTrue(
            os.path.exists(snapshot_dir),
            msg=f"精准Revision快照目录{snapshot_dir}不存在，--revision参数未生效"
        )
        print(f"✅ 精准revision参数生效：{self.revision[:8]}（完整hash：{self.revision}）")


if __name__ == "__main__":
    unittest.main()
