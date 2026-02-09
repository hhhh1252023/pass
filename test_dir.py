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
# 修复：适配0.36.0版本的导入（移除HfHubHTTPError，改用HuggingFaceHubError）
from huggingface_hub import HfApi, snapshot_download, HuggingFaceHubError

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)


class TestDownloadDir(CustomTestCase):
    """Testcase：Verify set --download-dir and --revision parameters take effect, inference request is successful.

       [Test Category] Parameter
       [Test Target] --download-dir, --revision
       """
    model = "microsoft/Phi-4-multimodal-instruct"
    revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
    download_dir = "./phi4_multimodal_weight"

    @classmethod
    def setUpClass(cls):
        # 第一步：适配0.36.0版本的revision验证（用get_commit_info，因为0.36.0版本支持）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        api = HfApi()
        
        try:
            # 0.36.0版本支持get_commit_info，直接用这个方法（最简洁）
            commit_info = api.get_commit_info(
                repo_id=cls.model,
                commit_hash=cls.revision
            )
            print(f"✅ Revision验证通过：{cls.revision}")
            print(f"版本提交时间：{commit_info.commit_time}")
            print(f"版本提交说明：{commit_info.commit_message}")
        
        except HuggingFaceHubError as e:
            # 捕获通用的Hub异常（替代原HfHubHTTPError）
            raise RuntimeError(f"❌ Revision {cls.revision} 无效：{e}")
        except Exception as e:
            raise RuntimeError(f"❌ Revision验证失败：{e}")

        # 第二步：创建下载目录
        run_command(f"mkdir -p {cls.download_dir}")
        
        # 第三步：启动服务（新增--revision参数）
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
        kill_process_tree(cls.process.pid)
        run_command(f"rm -rf {cls.download_dir}")

    def test_download_dir_and_revision(self):
        # 发送推理请求（适配多模态模型，延长超时）
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
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        # 验证--download-dir生效
        weight_suffixes = ("*.safetensors", "*.bin", "*.pth")
        weight_files = []
        for suffix in weight_suffixes:
            weight_files.extend(glob.glob(os.path.join(self.download_dir, "**", suffix), recursive=True))
        self.assertGreater(
            len(weight_files),
            0,
            msg=f"--download-dir {self.download_dir} 无模型权重文件"
        )

        # 验证revision对应的版本文件存在
        config_file = os.path.join(self.download_dir, "config.json")
        self.assertTrue(
            os.path.exists(config_file),
            msg=f"版本配置文件{config_file}不存在，revision可能未生效"
        )
        print(f"✅ 版本配置文件验证通过：{config_file} 存在")


if __name__ == "__main__":
    unittest.main()
