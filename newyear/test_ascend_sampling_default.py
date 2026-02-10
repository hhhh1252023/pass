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
    # 改为当前目录（运行脚本的目录）
    "metrics_dir": os.path.abspath("."),
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
        # 确保参数传递正确（当前目录）
        "--export-metrics-to-file",
        "--export-metrics-to-file-dir", os.path.abspath("."),
    ],
    "request_timeout": 60
}


class BaseSamplingTest(CustomTestCase):
    """基础测试类：封装通用逻辑"""
    server_process = None
    model_gen_config = None

    @classmethod
    def setUpClass(cls):
        """类级别初始化：仅执行一次"""
        # 1. 确认当前目录可写
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        # 验证目录可写
        test_file = metrics_dir / "test_write_perm.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"✅ 当前目录可写：{metrics_dir}")
        except PermissionError:
            raise RuntimeError(f"❌ 当前目录无写入权限：{metrics_dir}")
        
        # 2. 读取模型配置
        cls.model_gen_config = cls._load_model_gen_config()
        
        # 3. 启动服务
        cls._launch_server()
        
        print(f"\n=== {cls.__name__} 初始化完成 ===")
        print(f"模型配置默认参数：{cls.model_gen_config}")
        print(f"SGLang内置默认参数：{COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']}")
        # 打印当前目录文件，确认初始状态
        print(f"📂 当前目录初始文件：{[f.name for f in metrics_dir.glob('*') if f.is_file()]}")

    @classmethod
    def tearDownClass(cls):
        """类级别清理：仅关闭一次服务"""
        if cls.server_process:
            kill_process_tree(cls.server_process.pid)
            time.sleep(1)
            print(f"\n=== {cls.__name__} 服务已关闭 ===")
            # 打印最终目录文件，便于查看生成的metrics
            metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
            print(f"📂 测试完成后当前目录文件：{[f.name for f in metrics_dir.glob('*') if f.is_file()]}")

    def setUp(self):
        """每个测试方法前：不删除日志文件，仅打印当前文件列表"""
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        # 注释删除逻辑，保留日志文件
        # for file in metrics_dir.glob("*"):
        #     if file.is_file():
        #         file.unlink()
        time.sleep(0.5)
        # 打印当前目录文件，便于排查
        print(f"\n📂 测试方法执行前目录文件：{[f.name for f in metrics_dir.glob('*') if f.is_file()]}")

    @classmethod
    def _load_model_gen_config(cls):
        """读取模型generation_config.json"""
        gen_config_path = Path(COMMON_CONFIG["model"]) / "generation_config.json"
        if not gen_config_path.exists():
            raise FileNotFoundError(f"模型配置文件不存在：{gen_config_path}")
        
        with open(gen_config_path, "r", encoding="utf-8") as f:
            gen_config = json.load(f)
        
        core_params = {}
        for key in COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"].keys():
            core_params[key] = gen_config.get(key, COMMON_CONFIG["SGLANG_BUILTIN_DEFAULTS"][key])
        return core_params

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
        
        # 延长日志写入等待时间（从1秒改为3秒）
        time.sleep(3)
        return self._get_sampling_params_from_metrics()

    def _get_sampling_params_from_metrics(self):
        """提取metrics中的采样参数（增加详细打印）"""
        metrics_dir = Path(COMMON_CONFIG["metrics_dir"])
        # 打印当前目录所有文件，便于排查
        all_files = [f.name for f in metrics_dir.glob("*") if f.is_file()]
        print(f"\n🔍 提取参数时目录文件：{all_files}")
        
        # 匹配所有metrics相关文件（兼容不同命名格式）
        metrics_files = list(metrics_dir.glob("metrics-*.log")) + list(metrics_dir.glob("*.metrics"))
        print(f"🔍 匹配到的metrics文件：{[f.name for f in metrics_files]}")
        
        if not metrics_files:
            # 不直接断言失败，先打印详细信息再失败，便于排查
            self.fail(f"❌ 未找到metrics日志文件！当前目录文件：{all_files}")
        
        latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
        print(f"🔍 读取最新metrics文件：{latest_file.name}")
        
        sampling_params = {}
        with open(latest_file, "r", encoding="utf-8") as f:
            log_content = f.read()
            # 打印日志内容（便于调试）
            print(f"\n📝 metrics文件内容：\n{log_content[:500]}...")  # 只打印前500字符
            
            # 重新定位到文件开头解析
            f.seek(0)
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
        print(f"🔍 提取的采样参数：{core_params}")
        return core_params


class TestSamplingDefaultsModel(BaseSamplingTest):
    """测试 --sampling-defaults=model 模式"""
    @classmethod
    def _launch_server(cls):
        """启动model模式服务（仅依赖popen_launch_server）"""
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
        
        print(f"预期参数（模型配置）：{self.model_gen_config}")
        print(f"实际参数（metrics）：{sampling_params}")
        
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
        
        print(f"预期参数（SGLang内置）：{COMMON_CONFIG['SGLANG_BUILTIN_DEFAULTS']}")
        print(f"实际参数（metrics）：{sampling_params}")
        
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
        
        for key, value in custom_params.items():
            self.assertEqual(
                sampling_params[key], value,
                f"openai手工参数不生效：{key} 预期={value}, 实际={sampling_params[key]}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
