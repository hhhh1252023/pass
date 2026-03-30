"""Test loading weights from remote instance (Clean Version - Server Only).
Backend: transfer_engine
"""

import gc
import os
import unittest

import numpy as np
import requests
import torch
import torch.multiprocessing as mp

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import terminate_process
DEFAULT_SMALL_MODEL_NAME_FOR_TEST="/home/weights/LLM-Research/Llama-3.2-1B-Instruct/"
DEFAULT_URL_FOR_TEST="http://127.0.0.1:8234"

# 强制NPU多进程启动方式
mp.set_start_method("spawn", force=True)


def verify_params_close(params1, params2, error_msg):
    """Verify if two parameter arrays are close enough."""
    try:
        assert np.allclose(np.array(params1), np.array(params2)), error_msg
    except Exception as e:
        print(f"Parameters not close for {error_msg}")
        print("Params1:", np.array(params1))
        print("Params2:", np.array(params2))
        raise e


def init_process(
    rank,
    param_queue,
    truncate_size,
    tp_size,
    model_name,
    checking_parameters,
    seed_instance_ip,
    seed_instance_service_port,
    seed_instance_group_base_port,
    event_seed_ready,
    event_dst_ready_list,
    remote_instance_loader_backend,
):
    torch.npu.set_device(rank)

    if rank == 0:
        init_process_seed(
            rank,
            param_queue,
            truncate_size,
            model_name,
            checking_parameters,
            tp_size,
            event_seed_ready,
            event_dst_ready_list,
        )
    elif rank in [1, 2]:
        init_process_dst(
            rank,
            param_queue,
            truncate_size,
            model_name,
            seed_instance_ip,
            seed_instance_service_port,
            seed_instance_group_base_port,
            checking_parameters,
            tp_size,
            event_seed_ready,
            event_dst_ready_list,
            remote_instance_loader_backend,
        )


def init_process_seed(
    rank,
    param_queue,
    truncate_size,
    model_name,
    checking_parameters,
    tp_size,
    event_seed_ready,
    event_dst_ready_list,
):
    # NPU 关键环境变量
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    torch.npu.set_device(rank)
    torch.npu.synchronize()

    url = DEFAULT_URL_FOR_TEST
    # 启动种子实例 Server
    process = popen_launch_server(
        model_name,
        url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=(
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--base-gpu-id",
            12,
            "--tp-size",
            str(tp_size),
        ),
    )
    torch.npu.synchronize()

    # 获取种子实例权重
    seed_params = []
    for parameter_name in checking_parameters:
        seed_params.append(
            requests.get(
                f"{url}/get_weights_by_name",
                json={
                    "name": parameter_name,
                    "truncate_size": truncate_size,
                },
            ).json()
        )
    param_queue.put((f"seed_params", seed_params))

    # 同步：通知目标实例种子就绪
    event_seed_ready.set()
    # 等待所有目标实例完成
    for i in range(len(event_dst_ready_list)):
        event_dst_ready_list[i].wait()
    terminate_process(process)


def init_process_dst(
    rank,
    param_queue,
    truncate_size,
    model_name,
    seed_instance_ip,
    seed_instance_service_port,
    seed_instance_group_base_port,
    checking_parameters,
    tp_size,
    event_seed_ready,
    event_dst_ready_list,
    remote_instance_loader_backend,
):
    torch.npu.set_device(rank * tp_size)
    torch.npu.synchronize()
    base_gpu_id = rank * tp_size

    # 等待种子实例就绪
    event_seed_ready.wait()
    # 串行加载：等待前一个目标实例
    for i in range(rank - 1):
        event_dst_ready_list[i].wait()

    # 分配通信端口
    ports = []
    for i in range(tp_size):
        ports.append(seed_instance_group_base_port + (rank - 1) * tp_size + i)

    # 仅保留 Server 模式
    host, _, port = DEFAULT_URL_FOR_TEST.rpartition(":")
    url = ":".join([host, str(int(port) + 10000 + rank)])

    print(f"[sgl] rank {rank} init server on url: {url}")
    process = popen_launch_server(
        model_name,
        url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=(
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--base-gpu-id",
            14,
            "--tp-size",
            str(tp_size),
            "--cuda-graph-max-bs",
            2,
            "--tokenizer-path",
            model_name,
            "--remote-instance-weight-loader-seed-instance-ip",
            seed_instance_ip,
            "--remote-instance-weight-loader-seed-instance-service-port",
            seed_instance_service_port,
            "--remote-instance-weight-loader-send-weights-group-ports",
            f"[{','.join(str(port) for port in ports)}]",
            "--load-format",
            "remote_instance",
            # ✅ 核心：使用 transfer_engine 后端
            "--remote-instance-weight-loader-backend",
            remote_instance_loader_backend,
        ),
    )
    torch.npu.synchronize()

    # 标记当前目标实例就绪
    event_dst_ready_list[rank - 1].set()

    # 获取目标实例加载后的权重
    dst_params = []
    for parameter_name in checking_parameters:
        dst_params.append(
            requests.get(
                f"{url}/get_weights_by_name",
                json={"name": parameter_name, "truncate_size": truncate_size},
            ).json()
        )

    param_queue.put((f"sgl_dp_{rank}_dst_params", dst_params))

    # 关闭服务
    terminate_process(process)


def test_load_weights_from_remote_instance(
    tp_size,
    dp_size,
    model_name,
    truncate_size,
    checking_parameters,
    seed_instance_ip,
    seed_instance_service_port,
    seed_instance_group_base_port,
    remote_instance_loader_backend,
):
    print(
        f"Testing model: {model_name} | tp_size: {tp_size} | dp_size: {dp_size} | backend: transfer_engine"
    )
    param_queue = mp.Queue()
    results = {}
    event_seed_ready = mp.Event()
    event_dst_ready_list = [mp.Event() for _ in range(dp_size)]

    # 启动多进程：1个种子 + dp个目标
    context = mp.spawn(
        init_process,
        args=(
            param_queue,
            truncate_size,
            tp_size,
            model_name,
            checking_parameters,
            seed_instance_ip,
            seed_instance_service_port,
            seed_instance_group_base_port,
            event_seed_ready,
            event_dst_ready_list,
            remote_instance_loader_backend,
        ),
        nprocs=1 + dp_size,
        join=False,
    )

    # 收集权重数据
    while len(results) < (1 + dp_size):
        try:
            key, value = param_queue.get(timeout=5)
            results[key] = value
        except Exception as e:
            if all(not p.is_alive() for p in context.processes):
                break

    context.join()

    if len(results) != (1 + dp_size):
        raise RuntimeError(
            f"Expected {(1 + dp_size)} parameters but got {len(results)}"
        )

    # 整理权重数据
    params = {
        "seed": results.get("seed_params"),
        "sgl_dp_1_dest": results.get("sgl_dp_1_dst_params"),
    }
    if dp_size == 2:
        params["sgl_dp_2_dest"] = results.get("sgl_dp_2_dst_params")

    # 核心校验：权重一致性
    for i in range(len(params["seed"])):
        verify_params_close(
            params["seed"][i],
            params["sgl_dp_1_dest"][i],
            f"sgl_dp_1_dst_params rank {i}",
        )
        if dp_size == 2:
            verify_params_close(
                params["seed"][i],
                params["sgl_dp_2_dest"][i],
                f"sgl_dp_2_dst_params rank {i}",
            )

    # 资源清理
    del context
    param_queue.close()
    param_queue.join_thread()
    gc.collect()
    torch.npu.empty_cache()


class TestLoadWeightsFromRemoteInstance(CustomTestCase):
    def test_load_weights_from_remote_instance(self):
        # 硬件检查：至少2张NPU
        assert torch.npu.device_count() >= 2, "At least 2 NPUs are required"

        # ===================== 唯一修改点 =====================
        # 测试用例：后端改为 transfer_engine
        test_suits = [
            (1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "transfer_engine"),
            (1, 1, DEFAULT_SMALL_MODEL_NAME_FOR_TEST, "nccl"),
        ]
        # ======================================================

        # 权重校验配置
        truncate_size = 10
        checking_parameters = [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.2.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.4.self_attn.o_proj.weight",
            "model.layers.5.mlp.gate_proj.weight",
            "model.layers.6.mlp.up_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
            "model.layers.8.post_attention_layernorm.weight",
            "model.norm.weight",
        ]

        # 执行测试
        for tp_size, dp_size, model_name, remote_backend in test_suits:
            test_load_weights_from_remote_instance(
                tp_size,
                dp_size,
                model_name,
                truncate_size,
                checking_parameters,
                "127.0.0.1",
                8234,
                60010,
                remote_backend
            )


if __name__ == "__main__":
    unittest.main()
