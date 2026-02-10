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


# 通用配置抽离
COMMON_CONFIG = {
    "model": "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B",
    "base_url": DEFAULT_URL_FOR_TEST,
    "metrics_dir": "/tmp/sglang_metrics_test",
    "SGLANG_BUILTIN_DEFAULTS": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
    "base_server_args": [
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static", 0.8,
        "--tp-size", 2,
        "--export-metrics-to-file",
        "--export-metrics-to-file-dir", "/tmp/sglang_metrics_test",
    ]
}


class BaseSamplingTest(CustomTestCase):
    """基础测试类：封装通用逻辑"""
    server_process = None
    model_gen_config = None

    @classmethod
    def setUpClass(cls):
        """类级别初始化：仅执行一次"""
        # 1. 创建metrics目录
        Path(COMMON_CONFIG["metrics_dir"]).mkdir(parents=True, exist_ok=True)
        
        # 2. 读取模型generation_config.json
        cls.model_gen_config = cls._load_model_gen_config()
        
        # 3. 启动对应模式的服务（信任popen_launch_server的完备性）
        cls._launch_server()
        
        print(f"\n=== {cls.__name__} 初始化完成 ===")
        print(f"模型配置默认参数：{cls.model_gen_config}")
        print(f"SGLang内置默认参数：{COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']}")

    @classmethod
    def tearDownClass(cls):
        """类级别清理：仅关闭一次服务"""
        if cls.server_process:
            kill_process_tree(cls.server_process.pid)
            time.sleep(1)
            print(f"\n=== {cls.__name__} 服务已关闭 ===")

    def setUp(self):
        """每个测试方法前：清空metrics日志"""
        for file in Path(COMMON_CONFIG["metrics_dir"]).glob("*"):
            if file.is_file():
                file.unlink()
        time.sleep(0.5)  # 确保日志清空完成

    @classmethod
    def _load_model_gen_config(cls):
        """读取模型generation_config.json"""
        gen_config_path = Path(COMMON_CONFIG["model"]) / "generation_config.json"
        if not gen_config_path.exists():
            raise FileNotFoundError(f"模型配置文件不存在：{gen_config_path}")
        
        with open(gen_config_path, "r", encoding="utf-8") as f:
            gen_config = json.load(f)
        
        # 提取核心采样参数
        core_params = {}
        for key in COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"].keys():
            core_params[key] = gen_config.get(key, COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"][key])
        return core_params

    @classmethod
    def _launch_server(cls):
        """启动服务（子类实现具体逻辑）"""
        raise NotImplementedError("子类必须实现_launch_server方法")

    def _call_chat(self, custom_params: dict = None):
        """调用接口（无健康检查，信任popen启动的服务）"""
        req_body = {
            "model": COMMON_CONFIG["model"],
            "messages": [{"role": "user", "content": "测试采样参数：1+1=？"}]
        }
        if custom_params:
            req_body.update(custom_params)
        
        # 直接调用接口（信任popen_launch_server已确保服务就绪）
        response = requests.post(
            f"{COMMON_CONFIG['base_url']}/v1/chat/completions",
            json=req_body,
            timeout=10
        )
        self.assertEqual(response.status_code, 200, f"接口调用失败：{response.text}")
        
        # 等待metrics日志写入
        time.sleep(1)
        return self._get_sampling_params_from_metrics()

    def _get_sampling_params_from_metrics(self):
        """提取metrics中的采样参数"""
        metrics_files = list(Path(COMMON_CONFIG["metrics_dir"]).glob("metrics-*.log"))
        if not metrics_files:
            self.fail("未找到metrics日志文件")
        
        latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
        sampling_params = {}
        with open(latest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log_data = json.loads(line)
                    if "sampling_params" in log_data:
                        sampling_params = log_data["sampling_params"]
                        break
                except json.JSONDecodeError:
                    continue
        
        # 提取核心参数
        core_params = {}
        for key in COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"].keys():
            core_params[key] = sampling_params.get(key)
        return core_params


class TestSamplingDefaultsModel(BaseSamplingTest):
    """测试 --sampling-defaults=model 模式"""
    @classmethod
    def _launch_server(cls):
        """启动model模式服务（仅依赖popen_launch_server）"""
        server_args = COMMON_CONFIG["base_server_args"] + ["--sampling-defaults", "model"]
        print(f"\n=== 启动model模式服务 ===")
        print(f"启动参数：{server_args}")
        
        # 直接调用popen_launch_server（信任其超时和就绪逻辑）
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
        
        # 打印对比
        print(f"预期参数（模型配置）：{self.model_gen_config}")
        print(f"实际参数（metrics）：{sampling_params}")
        
        # 验证
        for key in COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"].keys():
            self.assertEqual(
                sampling_params[key], self.model_gen_config[key],
                f"model默认参数不匹配：{key} 预期={self.model_gen_config[key]}, 实际={sampling_params[key]}"
            )

    def test_custom_params(self):
        """model模式 - 手工自定义参数"""
        print("\n=== 测试model模式手工参数 ===")
        custom_params = {
            "temperature": 0.6,
            "top_p": 0.75,
            "top_k": 100,
            "min_p": 0.2,
            "repetition_penalty": 1.1
        }
        print(f"手工配置参数：{custom_params}")
        
        sampling_params = self._call_chat(custom_params)
        print(f"实际参数（metrics）：{sampling_params}")
        
        # 验证
        for key, value in custom_params.items():
            self.assertEqual(
                sampling_params[key], value,
                f"model手工参数不生效：{key} 预期={value}, 实际={sampling_params[key]}"
            )


class TestSamplingDefaultsOpenAI(BaseSamplingTest):
    """测试 --sampling-defaults=openai 模式"""
    @classmethod
    def _launch_server(cls):
        """启动openai模式服务（仅依赖popen_launch_server）"""
        server_args = COMMON_CONFIG["base_server_args"] + ["--sampling-defaults", "openai"]
        print(f"\n=== 启动openai模式服务 ===")
        print(f"启动参数：{server_args}")
        
        # 直接调用popen_launch_server
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
        
        # 打印对比
        print(f"预期参数（SGLang内置）：{COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']}")
        print(f"实际参数（metrics）：{sampling_params}")
        
        # 验证
        for key in COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"].keys():
            self.assertEqual(
                sampling_params[key], COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"][key],
                f"openai默认参数不匹配：{key} 预期={COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS'][key]}, 实际={sampling_params[key]}"
            )

    def test_custom_params(self):
        """openai模式 - 手工自定义参数"""
        print("\n=== 测试openai模式手工参数 ===")
        custom_params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.1,
            "repetition_penalty": 1.3
        }
        print(f"手工配置参数：{custom_params}")
        
        sampling_params = self._call_chat(custom_params)
        print(f"实际参数（metrics）：{sampling_params}")
        
        # 验证
        for key, value in custom_params.items():
            self.assertEqual(
                sampling_params[key], value,
                f"openai手工参数不生效：{key} 预期={value}, 实际={sampling_params[key]}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
