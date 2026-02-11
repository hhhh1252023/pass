import json
import requests
import unittest
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


# 通用配置抽离
COMMON_CONFIG = {
    "model": "/root/.cache/modelscope/hub/models/Qwen/Qwen3-32B",
    "accuracy": 0.89,
    "base_args": [
        "--trust-remote-code",
        "--mem-fraction-static", "0.8",
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
        "--tp-size", "4",
        "--disable-radix-cache",
        "--chunked-prefill-size", "-1",
    ],
    "request_timeout": 120
}


class TestScoreWithDelimiter(CustomTestCase):
    """测试类1：开启 --multi-item-scoring-delimiter"""
    server_process = None

    @classmethod
    def setUpClass(cls):
        """类级别初始化：启动带delimiter的服务（信任popen_launch_server）"""
        print("\n=== 初始化【开启delimiter】测试环境 ===")
        # 构造完整启动参数
        server_args = COMMON_CONFIG["base_args"] + [
            "--multi-item-scoring-delimiter", "151643"
        ]
        # 信任popen_launch_server的就绪逻辑，无需sleep
        cls.server_process = popen_launch_server(
            model=COMMON_CONFIG["model"],
            url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args
        )
        print(f"✅ 服务启动完成（带delimiter），参数：{server_args}")

    @classmethod
    def tearDownClass(cls):
        """类级别清理：仅关闭服务"""
        if cls.server_process:
            kill_process_tree(cls.server_process.pid)
            print("\n=== 【开启delimiter】测试服务已关闭 ===")

    def test_score_logic_with_delimiter(self):
        """验证开启delimiter后的分数大小逻辑"""
        print("\n=== 测试：开启 --multi-item-scoring-delimiter ===")
        # 构造score接口请求
        req_data = {
            "query": "Is this the correct result of 1 plus 2? ",
            "items": ["It is 3", "It is 4", "It is 5"],
            "label_token_ids": [9454, 2753],
            "apply_softmax": True,
            "item_first": False
        }

        # 调用接口（信任popen启动的服务已就绪）
        response = requests.post(
            url=f"{DEFAULT_URL_FOR_TEST}/v1/score",
            json=req_data,
            headers={"Content-Type": "application/json"},
            timeout=COMMON_CONFIG["request_timeout"]
        )
        self.assertEqual(response.status_code, 200, 
                         f"❌ 接口返回状态码错误：预期200，实际{response.status_code}")
        
        # 解析结果并验证逻辑
        result = response.json()
        scores = result["scores"]
        print(f"📝 接口返回scores：{scores}")

        # 核心逻辑断言：仅正确项满足score[0]>score[1]
        self.assertTrue(scores[0][0] > scores[0][1], 
                        "❌ 正确项（It is 3）分数逻辑错误：score[0] 应大于 score[1]")
        self.assertTrue(scores[1][0] < scores[1][1], 
                        "❌ 错误项（It is 4）分数逻辑错误：score[0] 应小于 score[1]")
        self.assertTrue(scores[2][0] < scores[2][1], 
                        "❌ 错误项（It is 5）分数逻辑错误：score[0] 应小于 score[1]")
        print("✅ 开启delimiter：分数逻辑验证通过！")


class TestScoreWithoutDelimiter(CustomTestCase):
    """测试类2：关闭 --multi-item-scoring-delimiter"""
    server_process = None

    @classmethod
    def setUpClass(cls):
        """类级别初始化：启动不带delimiter的服务（信任popen_launch_server）"""
        print("\n=== 初始化【关闭delimiter】测试环境 ===")
        # 直接使用基础参数（不含delimiter）
        server_args = COMMON_CONFIG["base_args"].copy()
        # 信任popen_launch_server的就绪逻辑，无需sleep
        cls.server_process = popen_launch_server(
            model=COMMON_CONFIG["model"],
            url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args
        )
        print(f"✅ 服务启动完成（无delimiter），参数：{server_args}")

    @classmethod
    def tearDownClass(cls):
        """类级别清理：仅关闭服务"""
        if cls.server_process:
            kill_process_tree(cls.server_process.pid)
            print("\n=== 【关闭delimiter】测试服务已关闭 ===")

    def test_score_logic_without_delimiter(self):
        """验证关闭delimiter后的分数大小逻辑"""
        print("\n=== 测试：关闭 --multi-item-scoring-delimiter ===")
        # 构造score接口请求（与开启delimiter的请求完全一致）
        req_data = {
            "query": "Is this the correct result of 1 plus 2? ",
            "items": ["It is 3", "It is 4", "It is 5"],
            "label_token_ids": [9454, 2753],
            "apply_softmax": True,
            "item_first": False
        }

        # 调用接口（信任popen启动的服务已就绪）
        response = requests.post(
            url=f"{DEFAULT_URL_FOR_TEST}/v1/score",
            json=req_data,
            headers={"Content-Type": "application/json"},
            timeout=COMMON_CONFIG["request_timeout"]
        )
        self.assertEqual(response.status_code, 200, 
                         f"❌ 接口返回状态码错误：预期200，实际{response.status_code}")
        
        # 解析结果并验证逻辑
        result = response.json()
        scores = result["scores"]
        print(f"📝 接口返回scores：{scores}")

        # 核心逻辑断言：所有项都满足score[0]>score[1]
        self.assertTrue(scores[0][0] > scores[0][1], 
                        "❌ 正确项（It is 3）分数逻辑错误：score[0] 应大于 score[1]")
        self.assertTrue(scores[1][0] > scores[1][1], 
                        "❌ 错误项（It is 4）分数逻辑错误：score[0] 应大于 score[1]")
        self.assertTrue(scores[2][0] > scores[2][1], 
                        "❌ 错误项（It is 5）分数逻辑错误：score[0] 应大于 score[1]")
        print("✅ 关闭delimiter：分数逻辑验证通过！")


if __name__ == "__main__":
    # 运行两个独立测试类
    unittest.main(verbosity=2)
