import asyncio
import os
import re
import unittest
from typing import Any, List, Optional, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests_with_custom_params,
)

class TestPrioritySchedulingPreemptionThreshold(CustomTestCase):
    """验证 --priority-scheduling-preemption-threshold=5 的调度逻辑：执行顺序 C(10) > A(2) > B(5)"""
    
    @classmethod
    def setUpClass(cls):
        # 配置模型路径和基础URL
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        
        # 打开日志文件
        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")
        
        # 启动服务，核心配置优先级调度和抢占阈值
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests", "1",  # 单运行位，确保抢占逻辑可观测
                "--max-queued-requests", "10",  # 足够队列容量，避免请求被拒绝
                "--enable-priority-scheduling",  # 开启优先级调度（必需）
                "--priority-scheduling-preemption-threshold", "5",  # 配置抢占阈值5
                "--disable-cuda-graph",
                "--attention-backend", "ascend",  # 适配昇腾环境
                "--tp-size", "1",  # 单机单卡配置
                "--mem-fraction-static", "0.8"  # 昇腾内存配置
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
    
    @classmethod
    def tearDownClass(cls):
        # 清理进程和日志文件
        kill_process_tree(cls.process.pid)
        _verify_running_queued_requests(1, 10)  # 验证运行/排队请求数不超限
        cls.stdout.close()
        cls.stderr.close()
        if os.path.exists(STDOUT_FILENAME):
            os.remove(STDOUT_FILENAME)
        if os.path.exists(STDERR_FILENAME):
            os.remove(STDERR_FILENAME)
    
    def test_preemption_threshold_execution_order(self):
        """核心测试：提交A(2)→B(5)→C(10)，验证执行顺序 C>A>B 且所有请求成功"""
        # 步骤1：先提交作业A（优先级2），让其进入运行状态（长期占用运行位）
        request_a = {
            "priority": 2,
            "sampling_params": {"max_new_tokens": 500}  # 大token数，确保运行时间足够长
        }
        # 异步发送作业A，不等待完成（移除无效参数 delay_between_requests）
        loop = asyncio.get_event_loop()
        task_a = loop.create_task(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_a]
            )
        )
        # 等待1秒，确保作业A已启动并占用运行位
        asyncio.sleep(1)
        
        # 步骤2：提交作业B（5）和作业C（10）（移除无效参数 delay_between_requests）
        requests_b_c = [
            {"priority": 5, "sampling_params": {"max_new_tokens": 100}},  # 作业B
            {"priority": 10, "sampling_params": {"max_new_tokens": 100}}  # 作业C
        ]
        responses_b_c = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, requests_b_c
            )
        )
        
        # 步骤3：等待作业A完成，获取所有响应
        responses_a = loop.run_until_complete(task_a)
        # 合并所有响应（A + B + C）
        all_responses = responses_a + responses_b_c
        
        # 步骤4：验证所有请求均处理成功（状态码200，无错误）
        expected_status = [(200, None)] * 3  # 3个请求均成功
        e2e_latencies = []
        _verify_generate_responses(all_responses, expected_status, e2e_latencies)
        
        # 步骤5：验证执行顺序（C > A > B）
        # 提取各作业耗时（索引对应：0=A，1=B，2=C）
        latency_a = e2e_latencies[0]
        latency_b = e2e_latencies[1]
        latency_c = e2e_latencies[2]
        
        # 核心断言：C最先完成（耗时最短），其次是A，最后是B
        assert latency_c < latency_a < latency_b, \
            f"执行顺序不符合预期！预期 C<A<B，实际耗时：C={latency_c}, A={latency_a}, B={latency_b}"

# 辅助验证函数：验证响应状态并收集耗时
def _verify_generate_responses(
    responses: Tuple[int, Any, float],
    expected_code_and_error: Tuple[int, Any],
    e2e_latencies: List[Optional[float]],
):
    for got, expected in zip(responses, expected_code_and_error):
        got_status, got_json = got
        expected_status, expected_err = expected
        
        # 验证状态码200
        assert got_status == expected_status, \
            f"请求处理失败：预期状态码200，实际{got_status}，响应：{got_json}"
        
        # 验证无错误信息
        if got_status == 200:
            assert "error" not in got_json, f"请求返回错误信息：{got_json.get('error')}"
            # 收集端到端耗时
            e2e_latencies.append(got_json["meta_info"]["e2e_latency"])
        else:
            e2e_latencies.append(None)

# 辅助验证函数：验证运行/排队请求数不超过配置上限
def _verify_running_queued_requests(
    max_running_requests: int, max_queued_requests: int
):
    rr_pattern = re.compile(r"#running-req:\s*(\d+)")
    qr_pattern = re.compile(r"#queue-req:\s*(\d+)")
    
    if not os.path.exists(STDERR_FILENAME):
        return
    
    with open(STDERR_FILENAME, "r") as f:
        for line in f:
            rr_match = rr_pattern.search(line)
            if rr_match:
                assert int(rr_match.group(1)) <= max_running_requests, \
                    f"运行请求数超限：{rr_match.group(1)} > {max_running_requests}"
            qr_match = qr_pattern.search(line)
            if qr_match:
                assert int(qr_match.group(1)) <= max_queued_requests, \
                    f"排队请求数超限：{qr_match.group(1)} > {max_queued_requests}"

if __name__ == "__main__":
    unittest.main()
