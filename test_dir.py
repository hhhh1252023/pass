import unittest
import requests
import os
import glob
import subprocess
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
    """Testcase：Verify --download-dir and --revision take effect (wait for weight download).

       [Test Category] Parameter
       [Test Target] --download-dir, --revision
       """
    model = "microsoft/Phi-1.5"
    revision = "main"
    download_dir = "./phi1.5_weight"
    # 新增：权重下载超时时间（300秒=5分钟，足够下载2.84GB）
    DOWNLOAD_TIMEOUT = 300
    # 权重文件名称（Phi-1.5的权重文件名为model.safetensors）
    WEIGHT_FILE_NAME = "model.safetensors"

    @classmethod
    def setUpClass(cls):
        # 1. 配置国内镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # 2. 创建下载目录
        run_command(f"mkdir -p {cls.download_dir}")
        
        # 3. 启动服务
        other_args = [
            "--download-dir", cls.download_dir,
            "--revision", cls.revision,
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            cls.model, DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'process') and cls.process:
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.download_dir):
            run_command(f"rm -rf {cls.download_dir}")

    def wait_for_weight_download(self):
        """等待权重文件下载完成，带超时机制"""
        start_time = time.time()
        weight_found = False
        
        while time.time() - start_time < self.DOWNLOAD_TIMEOUT:
            # 递归查找权重文件
            weight_files = glob.glob(
                os.path.join(self.download_dir, "**", self.WEIGHT_FILE_NAME),
                recursive=True
            )
            if weight_files:
                weight_found = True
                print(f"✅ 权重文件下载完成：{weight_files[0]}")
                break
            # 每5秒检查一次，避免频繁查询
            time.sleep(5)
            elapsed = int(time.time() - start_time)
            print(f"等待权重下载中...已耗时{elapsed}秒（超时{self.DOWNLOAD_TIMEOUT}秒）")
        
        if not weight_found:
            raise TimeoutError(
                f"❌ 权重文件下载超时（{self.DOWNLOAD_TIMEOUT}秒），未找到 {self.WEIGHT_FILE_NAME}"
            )

    def test_download_dir_and_revision(self):
        # 第一步：等待权重下载完成（核心修复）
        self.wait_for_weight_download()

        # 第二步：发送推理请求（此时权重已加载，请求能正常响应）
        # 增加重试机制，避免服务刚加载完权重时响应延迟
        response = None
        for _ in range(3):
            try:
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    break
                time.sleep(5)
            except requests.exceptions.ConnectionError:
                time.sleep(5)
        
        self.assertIsNotNone(response, msg="推理请求多次重试仍失败")
        self.assertEqual(response.status_code, 200, msg="推理请求失败，HTTP状态码非200")
        self.assertIn("Paris", response.text, msg="推理结果错误，未包含'Paris'")

        # 第三步：验证--download-dir参数生效（此时权重文件已存在）
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

        # 第四步：验证--revision参数生效（检查快照目录）
        snapshot_dir = os.path.join(
            self.download_dir,
            f"models--{self.model.replace('/', '--')}",
            "snapshots"
        )
        # Phi-1.5的main分支会下载到具体的commit hash目录，只要快照目录存在即生效
        self.assertTrue(
            os.path.exists(snapshot_dir) and len(os.listdir(snapshot_dir)) > 0,
            msg=f"Revision快照目录{snapshot_dir}无效，--revision参数未生效"
        )
        print(f"✅ --revision参数生效，快照目录：{snapshot_dir}")


if __name__ == "__main__":
    unittest.main()
