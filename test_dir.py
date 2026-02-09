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
# 修复：仅导入必用的基础类，移除所有异常类导入（避免版本兼容问题）
from huggingface_hub import HfApi, snapshot_download

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
        # 第一步：版本验证（极简逻辑，无异常类依赖）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        api = HfApi()
        
        try:
            # 核心：用get_commit_info验证revision有效性（0.36.0支持）
            commit_info = api.get_commit_info(
                repo_id=cls.model,
                commit_hash=cls.revision
            )
            print(f"✅ Revision验证通过：{cls.revision}")
            print(f"版本提交时间：{commit_info.commit_time}")
            print(f"版本提交说明：{commit_info.commit_message}")
        
        except Exception as e:
            # 通用异常捕获（兼容所有版本）
            error_msg = f"❌ Revision {cls.revision} 无效或验证失败：{str(e)}"
            # 补充常见失败原因提示
            if "404" in str(e):
                error_msg += "\n提示：可能是commit hash错误，或模型仓库未同步该版本"
            raise RuntimeError(error_msg)

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
        # 清理进程和目录
        if hasattr(cls, 'process') and cls.process:
            kill_process_tree(cls.process.pid)
        if os.path.exists(cls.download_dir):
            run_command(f"rm -rf {cls.download_dir}")

    def test_download_dir_and_revision(self):
        # 1. 发送推理请求（适配多模态模型）
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
        self.assertEqual(response.status_code, 200, msg="推理请求失败，状态码非200")
        self.assertIn("Paris", response.text, msg="推理结果未包含预期内容'Paris'")

        # 2. 验证--download-dir生效（检查权重文件）
        weight_suffixes = ("*.safetensors", "*.bin", "*.pth")
        weight_files = []
        for suffix in weight_suffixes:
            weight_files.extend(glob.glob(os.path.join(self.download_dir, "**", suffix), recursive=True))
        self.assertGreater(
            len(weight_files),
            0,
            msg=f"--download-dir {self.download_dir} 未找到任何模型权重文件"
        )

        # 3. 验证revision版本配置文件存在
        config_file = os.path.join(self.download_dir, "config.json")
        self.assertTrue(
            os.path.exists(config_file),
            msg=f"版本配置文件{config_file}不存在，revision可能未生效"
        )
        print(f"✅ 版本配置文件验证通过：{config_file} 存在")


if __name__ == "__main__":
    unittest.main()
