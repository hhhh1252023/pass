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
# 新增：兼容低版本的导入
from huggingface_hub import HfApi, snapshot_download, HfHubHTTPError

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
        # 第一步：兼容低版本的revision验证逻辑（替代get_commit_info）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        api = HfApi()
        
        try:
            # 方法1：尝试用list_repo_commits验证（低版本兼容）
            # 列出指定commit的信息，验证哈希是否存在
            commits = api.list_repo_commits(
                repo_id=cls.model,
                commit_hash=cls.revision
            )
            # 转换为列表（generator需遍历）
            commit_list = list(commits)
            if len(commit_list) == 0:
                raise ValueError(f"Revision {cls.revision} not found in repo")
            
            # 提取版本信息（低版本返回的Commit对象字段）
            commit_info = commit_list[0]
            print(f"✅ Revision验证通过：{cls.revision}")
            print(f"版本提交时间：{commit_info.created_at}")
            print(f"版本提交说明：{commit_info.message}")
        
        except AttributeError:
            # 方法2：若list_repo_commits也不存在，用下载配置文件验证（终极兼容）
            try:
                # 仅下载配置文件，不下载权重，快速验证版本存在性
                snapshot_download(
                    repo_id=cls.model,
                    revision=cls.revision,
                    local_dir="./temp_verify_revision",
                    ignore_patterns=["*.bin", "*.safetensors", "*.pth"],  # 仅下配置
                    local_dir_use_symlinks=False
                )
                print(f"✅ Revision {cls.revision} 验证通过（配置文件下载成功）")
                # 清理临时目录
                run_command("rm -rf ./temp_verify_revision")
            except HfHubHTTPError as e:
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
