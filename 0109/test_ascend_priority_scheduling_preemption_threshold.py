import asyncio
import os
import re
import unittest
from typing import Any, List, Optional, Tuple, Dict

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

# 定义正则表达式，用于从prompt中提取request_id
REQUEST_ID_PATTERN = re.compile(r"\[REQUEST_ID:([A-C])\]")

class TestPrioritySchedulingPreemptionThreshold(CustomTestCase):
    """验证 --priority-scheduling-preemption-threshold=5 的调度逻辑：执行顺序 C(10) > A(2) > B(5)"""
    
    @classmethod
    def setUpClass(cls):
        # 配置模型路径（适配昇腾环境，替换为本地有效路径）
        cls.model = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        
        # 初始化日志文件
        cls.stdout = open(STDOUT_FILENAME, "w", encoding="utf-8")
        cls.stderr = open(STDERR_FILENAME, "w", encoding="utf-8")
        
        # 启动服务，核心配置优先级调度和抢占阈值（适配昇腾环境）
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests", "1",
                "--max-queued-requests", "10",
                "--enable-priority-scheduling",
                "--priority-scheduling-preemption-threshold", "5",
                "--disable-cuda-graph",
                "--attention-backend", "ascend",
                "--tp-size", "1",
                "--mem-fraction-static", "0.8",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
    
    @classmethod
    def tearDownClass(cls):
        # 安全清理进程
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        
        # 验证运行/排队请求数不超限
        _verify_running_queued_requests(1, 10)
        
        # 清理日志文件
        cls.stdout.close()
        cls.stderr.close()
        
        for filename in [STDOUT_FILENAME, STDERR_FILENAME]:
            if os.path.exists(filename):
                os.remove(filename)
    
    def test_preemption_threshold_execution_order(self):
        # 步骤1：定义3个请求，添加唯一标识（request_id），嵌入到prompt中
        request_configs = {
            "A": {"priority": 2, "max_new_tokens": 2000, "request_id": "A"},
            "B": {"priority": 5, "max_new_tokens": 100, "request_id": "B"},
            "C": {"priority": 10, "max_new_tokens": 100, "request_id": "C"},
        }
        
        # 构造请求参数：将request_id嵌入到prompt中（SGLang支持，响应会保留prompt）
        def build_request(request_id: str, priority: int, max_new_tokens: int):
            prompt = f"[REQUEST_ID:{request_id}] 请完成一个简单的文本生成任务（仅用于测试优先级调度，无需返回复杂内容）。"
            return {
                "priority": priority,
                "prompt": prompt,  # 核心：传递包含唯一标识的prompt
                "sampling_params": {"max_new_tokens": max_new_tokens}
            }
        
        request_a = build_request(
            request_configs["A"]["request_id"],
            request_configs["A"]["priority"],
            request_configs["A"]["max_new_tokens"]
        )
        request_b = build_request(
            request_configs["B"]["request_id"],
            request_configs["B"]["priority"],
            request_configs["B"]["max_new_tokens"]
        )
        request_c = build_request(
            request_configs["C"]["request_id"],
            request_configs["C"]["priority"],
            request_configs["C"]["max_new_tokens"]
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 异步发送作业A，不等待完成（占用运行位）
        task_a = loop.create_task(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_a]
            )
        )
        
        # 延长等待时间，确保作业A已完全启动并占用运行位（提升测试稳定性）
        loop.run_until_complete(asyncio.sleep(1.0))
        
        # 发送作业B（排队等待）
        responses_b = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_b]
            )
        )
        loop.run_until_complete(asyncio.sleep(0.5))
        
        # 发送作业C（抢占A的运行位）
        responses_c = loop.run_until_complete(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url, [request_c]
            )
        )
        
        # 步骤3：等待作业A完成，获取所有响应
        responses_a = loop.run_until_complete(task_a)
        
        # 收集所有响应
        all_responses = responses_a + responses_b + responses_c
        
        # 步骤4：验证所有请求均处理成功，并收集带唯一标识的耗时
        expected_status = [(200, None)] * len(all_responses)
        e2e_latencies: List[float] = []
        _verify_generate_responses(all_responses, expected_status, e2e_latencies)
        
        # 步骤5：从响应的prompt中提取request_id，关联原始请求与耗时
        request_latencies: Dict[str, float] = {}
        for resp, latency in zip(all_responses, e2e_latencies):
            _, resp_json = resp
            
            # 提取prompt字段（SGLang响应会保留请求的prompt）
            assert "prompt" in resp_json, f"响应缺少必要字段 'prompt'，响应内容：{resp_json}"
            prompt = resp_json["prompt"]
            
            # 用正则提取request_id
            match = REQUEST_ID_PATTERN.search(prompt)
            assert match, f"无法从prompt中提取request_id，prompt内容：{prompt}"
            request_id = match.group(1)
            
            # 验证提取的request_id合法性
            assert request_id in ["A", "B", "C"], f"无法识别的请求ID：{request_id}"
            request_latencies[request_id] = latency
        
        # 步骤6：验证执行顺序（C < A < B）
        latency_a = request_latencies["A"]
        latency_b = request_latencies["B"]
        latency_c = request_latencies["C"]
        
        assert latency_c < latency_a, f"C的耗时应小于A！实际：C={latency_c}, A={latency_a}"
        assert latency_a < latency_b, f"A的耗时应小于B！实际：A={latency_a}, B={latency_b}"
        assert latency_c < latency_a < latency_b, \
            f"执行顺序不符合预期！预期 C<A<B，实际耗时：C={latency_c}, A={latency_a}, B={latency_b}"
        
        # 关闭事件循环，消除 "unclosed event loop" 资源警告
        loop.close()

# 辅助验证函数：验证响应状态并收集端到端耗时（修复类型注解不匹配问题）
def _verify_generate_responses(
    responses: List[Tuple[int, Any]],
    expected_code_and_error: List[Tuple[int, Optional[Any]]],
    e2e_latencies: List[Optional[float]],
):
    e2e_latencies.clear()  # 清空列表，避免残留数据干扰
    assert len(responses) == len(expected_code_and_error), \
        f"响应数量与预期不符！实际{len(responses)}，预期{len(expected_code_and_error)}"
    
    for got, expected in zip(responses, expected_code_and_error):
        # 拆分响应数据（状态码 + 响应体）
        got_status, got_json = got
        expected_status, expected_err = expected
        
        # 验证状态码是否符合预期
        assert got_status == expected_status, \
            f"请求处理失败：预期状态码{expected_status}，实际{got_status}，响应：{got_json}"
        
        # 验证响应无错误信息（仅针对200状态）
        if got_status == 200:
            assert "error" not in got_json, f"请求返回错误信息：{got_json.get('error', '未知错误')}"
            
            # 验证并收集端到端耗时（确保字段存在）
            assert "meta_info" in got_json, "响应缺少必要字段 'meta_info'"
            assert "e2e_latency" in got_json["meta_info"], "响应缺少必要字段 'e2e_latency'"
            e2e_latencies.append(got_json["meta_info"]["e2e_latency"])
        else:
            e2e_latencies.append(None)

# 辅助验证函数：验证运行/排队请求数不超过配置上限
def _verify_running_queued_requests(
    max_running_requests: int, max_queued_requests: int
):
    # 定义日志匹配模式
    rr_pattern = re.compile(r"#running-req:\s*(\d+)")
    qr_pattern = re.compile(r"#queue-req:\s*(\d+)")
    
    # 若日志文件不存在，直接返回
    if not os.path.exists(STDERR_FILENAME):
        return
    
    with open(STDERR_FILENAME, "r", encoding="utf-8") as f:
        for line in f:
            # 验证运行请求数
            rr_match = rr_pattern.search(line)
            if rr_match:
                running_req_count = int(rr_match.group(1))
                assert running_req_count <= max_running_requests, \
                    f"运行请求数超限：当前{running_req_count} > 上限{max_running_requests}"
            
            # 验证排队请求数
            qr_match = qr_pattern.search(line)
            if qr_match:
                queued_req_count = int(qr_match.group(1))
                assert queued_req_count <= max_queued_requests, \
                    f"排队请求数超限：当前{queued_req_count} > 上限{max_queued_requests}"

if __name__ == "__main__":
    unittest.main()
