# NPU CI 执行监控
**生成时间**: 2026-07-28 08:04 UTC
**分析 Run 数**: 2

---

## [Run #29411683974](https://github.com/sgl-project/sglang/actions/runs/29411683974)
- **分支**: `fix-kv-cache-aiter-memory-allocation`
- **总耗时**: 84.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29411683974

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073760) |
| stage-b-test-16-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073807) |
| multimodal-gen-test-1-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073818) |
| multimodal-gen-test-2-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073835) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 84.1min | 环境问题 | 依赖的Blob文件不存在导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340074158) |

- **stage-b-test-4-npu-a3**: 作业 stage-b-test-4-npu-a3 在尝试访问 Azure Blob 存储时返回 BlobNotFound 错误，可能是依赖的测试数据或模型文件被删除或路径配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073760

- **stage-b-test-16-npu-a3**: 作业 stage-b-test-16-npu-a3 在尝试访问 Azure Blob 存储时返回 BlobNotFound 错误，可能是依赖的测试数据或模型文件被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073807

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure 存储中未找到，可能是 CI 依赖的模型或数据文件缺失或路径错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073818

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，可能是依赖的模型权重或数据文件未上传或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073835

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示Azure Blob存储中找不到指定的blob（BlobNotFound），可能是模型权重或数据文件被删除或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340074158

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073794) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073812) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073824) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073893) |


## [Run #29404997476](https://github.com/sgl-project/sglang/actions/runs/29404997476)
- **分支**: `codex/kimi-vlm-warmup`
- **总耗时**: 245.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29404997476

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志不完整，未显示测试失败的具体错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254336) |
| multimodal-gen-test-2-npu-a3 | 62.9min | 其他 | 作业日志不完整，未显示测试执行与失败信息，仅包含环境准备和Node版本警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254342) |
| stage-b-test-4-npu-a3 | 48.6min | 代码错误 | 测试用例 test_npu_llada2_mini.py 执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254461) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions环境准备和清理信息，未包含任何测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254336

- **multimodal-gen-test-2-npu-a3**: 日志仅包含GitHub Actions初始化、Node 20弃用警告及上传工件步骤，未提供任何测试运行、错误堆栈或退出码，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254342

- **stage-b-test-4-npu-a3**: 在5个NPU测试中，4个通过，1个失败。失败用例为 test_npu_llada2_mini.py，退出码1，耗时895秒，具体错误信息未在日志中显示，需进一步查看该测试的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254461

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254296) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254297) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254345) |
| stage-b-test-1-npu-a2 (1) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254357) |
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254376) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254851) |


---
*Auto-generated by npu_pr_monitor.py*