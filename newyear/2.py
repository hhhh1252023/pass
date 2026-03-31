from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=195, suite="stage-b-test-1-gpu-small")

import gc
import torch
import unittest

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)


def _check_param(engine, param_name, expect_values):
    """校验权重前5个值是否符合预期"""
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.002
    ), f"{actual_values=}"


class TestUpdateWeightsFromTensor(CustomTestCase):
    def test_update_weights_from_tensor_load_format_custom(self):
        """仅测试：自定义权重加载器 + tp=1 单卡场景"""
        # 自定义权重加载器路径
        custom_loader_name = (
            "sglang.srt.model_executor.model_runner._model_load_weights_direct"
        )
        
        # 启动引擎：固定 tp=1，传入自定义加载器
        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            tp_size=1,
            custom_weight_loader=[custom_loader_name],
        )

        # 待更新的权重：qkv_proj
        write_param_names = [
            f"model.layers.{i}.self_attn.qkv_proj.weight" for i in range(6, 16)
        ]
        # 待校验的权重：k_proj（qkv拆分后）
        read_param_names = [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(6, 16)
        ]

        # 校验原始权重
        _check_param(
            engine, read_param_names[0], [-0.0198, 0.0227, 0.0168, 0.0232, -0.0178]
        )

        # 构造全1.5的新权重张量
        new_tensor = torch.full((3072, 2048), 1.5)
        
        # 调用API：使用自定义加载器更新权重
        engine.update_weights_from_tensor(
            [(name, new_tensor.clone()) for name in write_param_names],
            load_format=custom_loader_name,
        )

        # 校验权重更新成功
        for read_param_name in read_param_names[:3]:
            _check_param(engine, read_param_name, [1.5] * 5)

        # 关闭引擎，释放资源
        engine.shutdown()

        del new_tensor
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
