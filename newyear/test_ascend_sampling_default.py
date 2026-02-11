import json
import os
import time
import requests
import unittest
from pathlib import Path
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


# 通用配置（基于实际日志调整）
COMMON_CONFIG = {
    "model": "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B",
    "base_url": DEFAULT_URL_FOR_TEST,
    "metrics_dir": os.path.abspath("."),
    # SGLang内置默认（匹配日志）
    "SGLANG_BUILTIN_DEFAULTS": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
    # 模型generation_config默认（匹配日志）
    "MODEL_GEN_DEFAULTS": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
    "base_server_args": [
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static", 0.8,
        "--tp-size", 2,
        "--export-metrics-to-file",
        "--export-metrics-to-file-dir", os.path.abspath("."),
    ],
    "request_timeout": 60
}


class BaseSamplingTest(CustomTestCase):
    """基础测试类：适配实际日志格式"""
    server_process = None

    @classmethod
    def setUpClass(cls):
        """类级别初始化：仅执行一次"""
        # 确认当前目录可写
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        test_file = metrics_dir / "test_write_perm.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"✅ 当前目录可写：{metrics_dir}")
        except PermissionError:
            raise RuntimeError(f"❌ 当前目录无写入权限：{metrics_dir}")
        
        # 启动服务
        cls._launch_server()
        
        print(f"\n=== {cls.__name__} 初始化完成 ===")
        print(f"模型默认参数：{COMMON_CONFIG['MODEL_GEN_DEFAULTS']}")
        print(f"SGLang内置默认参数：{COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']}")
        print(f"📂 当前目录初始文件：{[f.name for f in metrics_dir.glob('*') if f.is_file()]}")

    @classmethod
    def tearDownClass(cls):
        """类级别清理：仅关闭一次服务"""
        if cls.server_process:
            kill_process_tree(cls.server_process.pid)
            time.sleep(1)
            print(f"\n=== {cls.__name__} 服务已关闭 ===")
            metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
            print(f"📂 测试完成后当前目录文件：{[f.name for f in metrics_dir.glob('*') if f.is_file()]}")

    def setUp(self):
        """每个测试方法前：不删除日志文件，仅打印当前文件列表"""
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        time.sleep(0.5)
        print(f"\n📂 测试方法执行前目录文件：{[f.name for f in metrics_dir.glob('*') if f.is_file()]}")

    @classmethod
    def _launch_server(cls):
        """启动服务（子类实现具体逻辑）"""
        raise NotImplementedError("子类必须实现_launch_server方法")

    def _call_chat(self, custom_params: dict = None):
        """调用接口（仅调整超时时间，无重试）"""
        req_body = {
            "model": COMMON_CONFIG["model"],
            "messages": [{"role": "user", "content": "测试采样参数：1+1=？"}]
        }
        if custom_params:
            req_body.update(custom_params)
        
        # 调用接口
        response = requests.post(
            f"{COMMON_CONFIG['base_url']}/v1/chat/completions",
            json=req_body,
            timeout=COMMON_CONFIG["request_timeout"]
        )
        self.assertEqual(response.status_code, 200, f"接口调用失败：{response.text}")
        
        # 延长日志写入等待时间
        time.sleep(3)
        return self._get_sampling_params_from_metrics()

    def _get_sampling_params_from_metrics(self):
        """提取metrics中的采样参数（适配实际日志格式）"""
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        # 匹配实际的metrics文件命名：sglang-request-metrics-*
        metrics_files = list(metrics_dir.glob("sglang-request-metrics-*.log"))
        print(f"\n🔍 匹配到的metrics文件：{[f.name for f in metrics_files]}")
        
        if not metrics_files:
            self.fail(f"❌ 未找到sglang-request-metrics-*.log文件！当前目录文件：{[f.name for f in metrics_dir.glob('*') if f.is_file()]}")
        
        # 取最新的metrics文件
        latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
        print(f"🔍 读取最新metrics文件：{latest_file.name}")
        
        # 读取并清理日志内容（解决换行/空格问题）
        with open(latest_file, "r", encoding="utf-8") as f:
            log_content = f.read()
            # 按行分割（日志可能有多条JSON）
            log_lines = [line.strip() for line in log_content.split("\n") if line.strip()]
            # 取最后一条有效日志（最新请求）
            last_log = log_lines[-1] if log_lines else ""
            # 清理换行和多余空格
            clean_content = last_log.replace("\n", "").replace("  ", " ").strip()
            print(f"\n📝 清理后的最新日志内容：\n{clean_content[:800]}...")
        
        # 解析JSON（核心适配：request_parameters里嵌套sampling_params）
        try:
            # 解析外层JSON
            log_data = json.loads(clean_content)
            # 解析request_parameters字段（字符串转JSON）
            req_params = json.loads(log_data["request_parameters"])
            # 提取sampling_params
            sampling_params = req_params.get("sampling_params", {})
            print(f"🔍 解析出的sampling_params：{sampling_params}")
        except json.JSONDecodeError as e:
            self.fail(f"❌ JSON解析失败：{e}，原始内容：{clean_content[:500]}")
        
        # 提取核心采样参数（补全缺失的参数为默认值）
        core_params = {}
        for key in COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"].keys():
            core_params[key] = sampling_params.get(key, COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"][key])
        print(f"🔍 最终提取的核心采样参数：{core_params}")
        return core_params


class TestSamplingDefaultsModel(BaseSamplingTest):
    """测试 --sampling-defaults=model 模式"""
    @classmethod
    def _launch_server(cls):
        """启动model模式服务"""
        server_args = COMMON_CONFIG["base_server_args"] + ["--sampling-defaults", "model"]
        print(f"\n=== 启动model模式服务 ===")
        print(f"启动参数：{server_args}")
        
        cls.server_process = popen_launch_server(
            COMMON_CONFIG["model"],
            COMMON_CONFIG["base_url"],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )

    def test_default_params(self):
        """model模式 - 默认参数（无手工配置）"""
        print("\n=== 测试model模式默认参数 ===")
        sampling_params = self._call_chat()
        
        # 精准断言：匹配日志中的model默认参数
        self.assertEqual(
            sampling_params["temperature"], COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["temperature"],
            f"temperature不匹配：预期={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['temperature']}, 实际={sampling_params['temperature']}"
        )
        self.assertEqual(
            sampling_params["top_p"], COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["top_p"],
            f"top_p不匹配：预期={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['top_p']}, 实际={sampling_params['top_p']}"
        )
        self.assertEqual(
            sampling_params["top_k"], COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["top_k"],
            f"top_k不匹配：预期={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['top_k']}, 实际={sampling_params['top_k']}"
        )
        self.assertEqual(
            sampling_params["min_p"], COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["min_p"],
            f"min_p不匹配：预期={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['min_p']}, 实际={sampling_params['min_p']}"
        )
        self.assertEqual(
            sampling_params["repetition_penalty"], COMMON_CONFIG["MODEL_GEN_DEFAULTS"]["repetition_penalty"],
            f"repetition_penalty不匹配：预期={COMMON_CONFIG['MODEL_GEN_DEFAULTS']['repetition_penalty']}, 实际={sampling_params['repetition_penalty']}"
        )
        print("✅ model模式默认参数断言通过！")

    def test_custom_params(self):
        """model模式 - 手工自定义参数"""
        print("\n=== 测试model模式手工参数 ===")
        # 手工配置的参数（匹配日志中的实际值）
        custom_params = {
            "temperature": 0.6,
            "top_p": 0.75,
            "top_k": 100,
            "min_p": 0.2,
            "repetition_penalty": 1.1
        }
        print(f"手工配置参数：{custom_params}")
        sampling_params = self._call_chat(custom_params)
        
        # 精准断言：手工参数完全生效
        for key, expected_value in custom_params.items():
            self.assertEqual(
                sampling_params[key], expected_value,
                f"手工参数{key}不生效：预期={expected_value}, 实际={sampling_params[key]}"
            )
        print("✅ model模式手工参数断言通过！")


class TestSamplingDefaultsOpenAI(BaseSamplingTest):
    """测试 --sampling-defaults=openai 模式"""
    @classmethod
    def _launch_server(cls):
        """启动openai模式服务"""
        server_args = COMMON_CONFIG["base_server_args"] + ["--sampling-defaults", "openai"]
        print(f"\n=== 启动openai模式服务 ===")
        print(f"启动参数：{server_args}")
        
        cls.server_process = popen_launch_server(
            COMMON_CONFIG["model"],
            COMMON_CONFIG["base_url"],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )

    def test_default_params(self):
        """openai模式 - 默认参数（无手工配置）"""
        print("\n=== 测试openai模式默认参数 ===")
        sampling_params = self._call_chat()
        
        # 精准断言：匹配日志中的openai默认参数（与SGLang内置一致）
        self.assertEqual(
            sampling_params["temperature"], COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["temperature"],
            f"temperature不匹配：预期={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['temperature']}, 实际={sampling_params['temperature']}"
        )
        self.assertEqual(
            sampling_params["top_p"], COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["top_p"],
            f"top_p不匹配：预期={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['top_p']}, 实际={sampling_params['top_p']}"
        )
        self.assertEqual(
            sampling_params["top_k"], COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["top_k"],
            f"top_k不匹配：预期={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['top_k']}, 实际={sampling_params['top_k']}"
        )
        self.assertEqual(
            sampling_params["min_p"], COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["min_p"],
            f"min_p不匹配：预期={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['min_p']}, 实际={sampling_params['min_p']}"
        )
        self.assertEqual(
            sampling_params["repetition_penalty"], COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"]["repetition_penalty"],
            f"repetition_penalty不匹配：预期={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']['repetition_penalty']}, 实际={sampling_params['repetition_penalty']}"
        )
        print("✅ openai模式默认参数断言通过！")

    def test_custom_params(self):
        """openai模式 - 手工自定义参数"""
        print("\n=== 测试openai模式手工参数 ===")
        # 手工配置的参数（匹配日志中的实际值）
        custom_params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.1,
            "repetition_penalty": 1.3
        }
        print(f"手工配置参数：{custom_params}")
        sampling_params = self._call_chat(custom_params)
        
        # 精准断言：手工参数完全生效
        for key, expected_value in custom_params.items():
            self.assertEqual(
                sampling_params[key], expected_value,
                f"手工参数{key}不生效：预期={expected_value}, 实际={sampling_params[key]}"
            )
        print("✅ openai模式手工参数断言通过！")


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
