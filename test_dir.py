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
# 新增：导入huggingface_hub用于验证revision版本
from huggingface_hub import HfApi

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)  # 调整预估时间（多模态模型启动稍久）


class TestDownloadDir(CustomTestCase):
    """Testcase：Verify set --download-dir and --revision parameters take effect, inference request is successful.

       [Test Category] Parameter
       [Test Target] --download-dir, --revision
       """
    # 替换为目标模型
    model = "microsoft/Phi-4-multimodal-instruct"
    # 指定目标revision（commit hash）
    revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
    download_dir = "./phi4_multimodal_weight"

    @classmethod
    def setUpClass(cls):
        # 第一步：先验证revision版本是否有效（关键！避免启动服务时版本错误）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
        api = HfApi()
        try:
            # 验证revision哈希是否存在且有效
            commit_info = api.get_commit_info(
                repo_id=cls.model,
                commit_hash=cls.revision
            )
            print(f"✅ Revision验证通过：{cls.revision}")
            print(f"版本提交时间：{commit_info.commit_time}")
            print(f"版本提交说明：{commit_info.commit_message}")
        except Exception as e:
            raise RuntimeError(f"❌ Revision {cls.revision} 无效：{e}")

        # 第二步：创建下载目录
        run_command(f"mkdir -p {cls.download_dir}")
        
        # 第三步：启动服务的参数（新增--revision）
        other_args = [
            "--download-dir",
            cls.download_dir,
            "--revision",  # 新增：指定版本
            cls.revision,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        # 启动服务（多模态模型稍大，超时时间保持默认即可）
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        # 清理进程和目录
        kill_process_tree(cls.process.pid)
        run_command(f"rm -rf {cls.download_dir}")

    def test_download_dir_and_revision(self):
        # 1. 发送多模态推理请求（适配Phi-4-multimodal-instruct）
        # 注：多模态模型支持文本+图像，这里先测试纯文本推理（简化验证）
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "What is the capital of France? Answer in one word.",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
            timeout=60  # 多模态模型推理稍久，延长超时
        )
        # 验证请求成功
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

        # 2. 验证--download-dir生效（检查权重文件）
        weight_suffixes = ("*.safetensors", "*.bin", "*.pth")
        weight_files = []
        for suffix in weight_suffixes:
            weight_files.extend(glob.glob(os.path.join(self.download_dir, "**", suffix), recursive=True))
        self.assertGreater(
            len(weight_files),
            0,
            msg=f"--download-dir {self.download_dir} 无模型权重文件"
        )

        # 3. 额外验证：下载的权重对应指定revision（可选，增强版本确认）
        # 检查下载目录中的commit_hash文件（SGLang会保存版本哈希）
        commit_file = os.path.join(self.download_dir, ".git", "refs", "heads", "main")
        if os.path.exists(commit_file):
            with open(commit_file, "r") as f:
                local_hash = f.read().strip()
            # 验证本地下载的版本哈希与指定的revision一致（前7位匹配即可，完整哈希也可）
            self.assertTrue(
                local_hash.startswith(self.revision[:7]),
                msg=f"本地版本哈希{local_hash}与指定revision{self.revision}不匹配"
            )
            print(f"✅ 本地下载的版本哈希验证通过：{local_hash[:7]} == {self.revision[:7]}")


if __name__ == "__main__":
    unittest.main()
