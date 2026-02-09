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
# 仅保留最基础、全版本兼容的导入
from huggingface_hub import snapshot_download

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)


class TestDownloadDir(CustomTestCase):
    """Testcase：Verify set --download-dir and --revision parameters take effect, inference request is successful.

       [Test Category] Parameter
       [Test Target] --download-dir, --revision
       """
    model = "microsoft/Phi-4-multimodal-instruct"
    revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
    download_dir = "./phi4_multimodal_weight"
    temp_verify_dir = "./temp_phi4_verify"

    @classmethod
    def setUpClass(cls):
        # 第一步：版本验证（全版本兼容，移除所有不兼容参数）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        try:
            # 核心：仅下载配置文件验证revision，移除timeout参数（兼容所有版本）
            snapshot_download(
                repo_id=cls.model,
                revision=cls.revision,
                local_dir=cls.temp_verify_dir,
                ignore_patterns=["*.bin", "*.safetensors", "*.pth"],
                local_dir_use_symlinks=False
            )
            print(f"✅ Revision {cls.revision} 验证通过：该版本存在且可访问")
            
            # 清理临时目录
            if os.path.exists(cls.temp_verify_dir):
                run_command(f"rm -rf {cls.temp_verify_dir}")
        
        except Exception as e:
            error_msg = f"❌ Revision {cls.revision} 无效或无法访问：{str(e)}"
            if "404" in str(e) or "not found" in str(e).lower():
                error_msg += "\n常见原因：1. commit hash输入错误 2. 镜像站未同步 3. 模型仓库无此版本"
            raise RuntimeError(error_msg)

        # 第二步：创建正式下载目录
        run_command(f"mkdir -p {cls.download_dir}")
        
        # 第三步：启动SGLang服务
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
        for dir_path in [cls.download_dir, cls.temp_verify_dir]:
            if os.path.exists(dir_path):
                run_command(f"rm -rf {dir_path}")

    def test_download_dir_and_revision(self):
        # 1. 发送推理请求
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "What is the capital of France? Answer in one word.",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
            timeout=60
        )
        self.assertEqual(response.status_code, 200, msg="推理请求失败，HTTP状态码非200")
        self.assertIn("Paris", response.text, msg="推理结果错误，未包含'Paris'")

        # 2. 验证--download-dir生效
        weight_suffixes = ("*.safetensors", "*.bin", "*.pth")
        weight_files = []
        for suffix in weight_suffixes:
            weight_files.extend(glob.glob(os.path.join(self.download_dir, "**", suffix), recursive=True))
        self.assertGreater(
            len(weight_files),
            0,
            msg=f"--download-dir {self.download_dir} 未找到权重文件，参数未生效"
        )

        # 3. 验证revision配置文件存在
        config_file = os.path.join(self.download_dir, "config.json")
        self.assertTrue(
            os.path.exists(config_file),
            msg=f"版本配置文件{config_file}不存在，--revision未生效"
        )
        print(f"✅ 版本配置文件验证通过：{config_file}")


if __name__ == "__main__":
    unittest.main()
