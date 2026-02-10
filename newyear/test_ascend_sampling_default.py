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


class TestSamplingDefaultsMetrics(CustomTestCase):
    """测试采样参数规则：
    1. model模式：默认参数从generation_config.json读取，手工参数优先
    2. openai模式：默认参数用SGLang内置值，手工参数优先
    每个模式都测试「默认参数」和「手工参数」两个场景，并打印实际生效的默认值
    """
    
    # 基础配置
    model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B"
    base_url = DEFAULT_URL_FOR_TEST
    metrics_dir = "/tmp/sglang_metrics_test"  # 可读写目录
    # SGLang内置默认采样参数（与业务逻辑对齐）
    SGLANG_BUILTIN_DEFAULTS = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    
    # 基础服务启动参数
    base_server_args = [
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static", 0.8,
        "--tp-size", 2,
        "--export-metrics-to-file",
        "--export-metrics-to-file-dir", metrics_dir,
    ]

    @classmethod
    def setUpClass(cls):
        """初始化：创建metrics目录 + 读取模型配置文件"""
        # 创建metrics目录
        Path(cls.metrics_dir).mkdir(parents=True, exist_ok=True)
        # 读取model模式的默认配置（generation_config.json）
        cls.model_gen_config = cls._load_model_generation_config()
        print(f"\n=== 预加载配置 ===")
        print(f"1. 模型generation_config.json中的默认参数：{cls.model_gen_config}")
        print(f"2. SGLang内置默认参数：{cls.SGLANG_BUILTIN_DEFAULTS}")

    @classmethod
    def _load_model_generation_config(cls):
        """读取模型的generation_config.json，返回采样相关参数"""
        gen_config_path = Path(cls.model) / "generation_config.json"
        if not gen_config_path.exists():
            raise FileNotFoundError(f"未找到模型配置文件：{gen_config_path}")
        
        with open(gen_config_path, "r", encoding="utf-8") as f:
            gen_config = json.load(f)
        
        # 提取核心采样参数（与内置默认参数对齐）
        core_params = {}
        for key in cls.SGLANG_BUILTIN_DEFAULTS.keys():
            core_params[key] = gen_config.get(key, cls.SGLANG_BUILTIN_DEFAULTS[key])
        return core_params

    def setUp(self):
        """每个测试前：清空metrics日志 + 初始化服务进程"""
        # 清空历史metrics日志
        for file in Path(self.metrics_dir).glob("*"):
            if file.is_file():
                file.unlink()
        self.server_process = None

    def tearDown(self):
        """每个测试后：关闭服务进程"""
        if self.server_process:
            kill_process_tree(self.server_process.pid)
            time.sleep(1)

    def _launch_server(self, sampling_defaults: str):
        """启动服务，返回启动日志（便于排查）"""
        server_args = self.base_server_args + ["--sampling-defaults", sampling_defaults]
        print(f"\n=== 启动服务：sampling-defaults={sampling_defaults} ===")
        print(f"启动参数：{server_args}")
        
        self.server_process = popen_launch_server(
            self.model, self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )
        time.sleep(2)  # 等待初始化完成
        return self.server_process

    def _call_chat(self, custom_params: dict = None):
        """调用接口，返回响应和实际生效的采样参数"""
        # 基础请求体
        req_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": "测试采样参数：1+1=？"}]
        }
        # 添加手工参数（如果有）
        if custom_params:
            req_body.update(custom_params)
        
        # 调用接口
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=req_body,
            timeout=10
        )
        self.assertEqual(response.status_code, 200, f"接口调用失败：{response.text}")
        
        # 等待metrics日志写入，然后提取采样参数
        time.sleep(1)
        sampling_params = self._get_sampling_params_from_metrics()
        return response, sampling_params

    def _get_sampling_params_from_metrics(self):
        """从metrics日志提取采样参数，返回结构化字典"""
        metrics_files = list(Path(self.metrics_dir).glob("metrics-*.log"))
        if not metrics_files:
            print("⚠️ 未找到metrics日志文件")
            return {}
        
        # 读取最新的metrics文件
        latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
        sampling_params = {}
        with open(latest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log_data = json.loads(line)
                    # 提取采样参数（适配实际日志字段，可根据实际调整）
                    if "sampling_params" in log_data:
                        sampling_params = log_data["sampling_params"]
                        break
                except json.JSONDecodeError:
                    continue
        
        # 只保留核心采样参数（便于对比）
        core_params = {}
        for key in self.SGLANG_BUILTIN_DEFAULTS.keys():
            core_params[key] = sampling_params.get(key)
        return core_params

    # ------------------------------ 核心测试用例 ------------------------------
    def test_model_mode_default_params(self):
        """测试model模式 - 默认参数（无手工配置）"""
        print("\n==================== test_model_mode_default_params ====================")
        # 1. 启动model模式服务
        self._launch_server("model")
        
        # 2. 无手工参数调用接口
        _, sampling_params = self._call_chat()
        
        # 3. 打印实际生效的默认参数
        print(f"\n=== model模式 - 默认参数场景 ===")
        print(f"预期默认参数（来自generation_config.json）：{self.model_gen_config}")
        print(f"实际生效参数（来自metrics日志）：{sampling_params}")
        
        # 4. 验证参数匹配
        for key in self.SGLANG_BUILTIN_DEFAULTS.keys():
            self.assertEqual(
                sampling_params[key], self.model_gen_config[key],
                f"model模式默认参数不匹配：{key} 预期={self.model_gen_config[key]}, 实际={sampling_params[key]}"
            )

    def test_model_mode_custom_params(self):
        """测试model模式 - 手工自定义参数"""
        print("\n==================== test_model_mode_custom_params ====================")
        # 1. 启动model模式服务
        self._launch_server("model")
        
        # 2. 定义手工参数
        custom_params = {
            "temperature": 0.6,
            "top_p": 0.75,
            "top_k": 100,
            "min_p": 0.2,
            "repetition_penalty": 1.1
        }
        print(f"\n=== model模式 - 手工参数场景 ===")
        print(f"手工配置参数：{custom_params}")
        
        # 3. 带手工参数调用接口
        _, sampling_params = self._call_chat(custom_params)
        print(f"实际生效参数（来自metrics日志）：{sampling_params}")
        
        # 4. 验证手工参数生效
        for key, value in custom_params.items():
            self.assertEqual(
                sampling_params[key], value,
                f"model模式手工参数不生效：{key} 预期={value}, 实际={sampling_params[key]}"
            )

    def test_openai_mode_default_params(self):
        """测试openai模式 - 默认参数（无手工配置）"""
        print("\n==================== test_openai_mode_default_params ====================")
        # 1. 启动openai模式服务
        self._launch_server("openai")
        
        # 2. 无手工参数调用接口
        _, sampling_params = self._call_chat()
        
        # 3. 打印实际生效的默认参数
        print(f"\n=== openai模式 - 默认参数场景 ===")
        print(f"预期默认参数（SGLang内置）：{self.SGLANG_BUILTIN_DEFAULTS}")
        print(f"实际生效参数（来自metrics日志）：{sampling_params}")
        
        # 4. 验证参数匹配
        for key in self.SGLANG_BUILTIN_DEFAULTS.keys():
            self.assertEqual(
                sampling_params[key], self.SGLANG_BUILTIN_DEFAULTS[key],
                f"openai模式默认参数不匹配：{key} 预期={self.SGLANG_BUILTIN_DEFAULTS[key]}, 实际={sampling_params[key]}"
            )

    def test_openai_mode_custom_params(self):
        """测试openai模式 - 手工自定义参数"""
        print("\n==================== test_openai_mode_custom_params ====================")
        # 1. 启动openai模式服务
        self._launch_server("openai")
        
        # 2. 定义手工参数
        custom_params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.1,
            "repetition_penalty": 1.3
        }
        print(f"\n=== openai模式 - 手工参数场景 ===")
        print(f"手工配置参数：{custom_params}")
        
        # 3. 带手工参数调用接口
        _, sampling_params = self._call_chat(custom_params)
        print(f"实际生效参数（来自metrics日志）：{sampling_params}")
        
        # 4. 验证手工参数生效
        for key, value in custom_params.items():
            self.assertEqual(
                sampling_params[key], value,
                f"openai模式手工参数不生效：{key} 预期={value}, 实际={sampling_params[key]}"
            )


if __name__ == "__main__":
    # 运行测试，打印详细日志
    unittest.main(verbosity=2)
