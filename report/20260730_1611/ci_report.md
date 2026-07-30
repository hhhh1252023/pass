# NPU CI 执行监控
**生成时间**: 2026-07-30 08:11 UTC
**分析 Run 数**: 25

---

## [Run #30522699084](https://github.com/sgl-project/sglang/actions/runs/30522699084)
- **分支**: `main`
- **总耗时**: 5.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30522699084

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 1.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597134) |
| stage-a-unit-test-npu | 4.0min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597145) |
| multimodal-gen-test-1-npu-a3 | 3.0min | 其他 | 作业日志不完整，未显示测试执行和失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597160) |
| stage-b-test-2-npu-a2 (0) | 4.0min | 环境问题 | 自定义容器执行失败，自托管Runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597167) |
| multimodal-gen-test-2-npu-a3 | 1.3min | 环境问题 | 自定义容器执行失败，导致作业无法启动 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597168) |
| stage-b-test-1-npu-a2 (1) | 4.0min | 环境问题 | 自定义容器执行失败，可能是Kubernetes Pod或NPU环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597179) |
| stage-b-test-2-npu-a2 (1) | 4.1min | 环境问题 | 自定义容器执行失败，可能是Kubernetes Pod调度或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597180) |
| stage-b-test-4-npu-a3 | 4.3min | 环境问题 | 自定义容器执行失败，可能是容器环境或资源问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597190) |
| stage-b-test-1-npu-a2 (0) | 4.0min | 环境问题 | 自定义容器执行失败，可能是Kubernetes环境问题 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597207) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 0.8min | 其他 | 作业日志不完整，未显示测试执行和失败的具体错误信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597579) |

- **stage-b-test-16-npu-a3**: 在下载triton-ascend包时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，可能是容器环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597134

- **stage-a-unit-test-npu**: 日志显示执行自定义容器实现失败，提示联系自托管运行器管理员，同时Node 20被弃用警告，但核心错误是容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597145

- **multimodal-gen-test-1-npu-a3**: 日志仅包含环境准备和清理步骤，缺少测试运行及失败的具体输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597160

- **stage-b-test-2-npu-a2 (0)**: 作业在运行自定义容器实现时出错，错误信息为'Executing the custom container implementation failed'，提示联系自托管Runner管理员，属于环境配置或容器运行时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597167

- **multimodal-gen-test-2-npu-a3**: 自托管运行器在执行自定义容器时失败，错误信息提示jobPod未设置，可能是Kubernetes Pod未成功创建或配置错误，需联系管理员检查运行器环境。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597168

- **stage-b-test-1-npu-a2 (1)**: 日志显示`Executing the custom container implementation failed`，表明自定义容器实现执行失败，可能由于Kubernetes Pod配置错误、NPU驱动问题或资源不足导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597179

- **stage-b-test-2-npu-a2 (1)**: 日志显示在运行测试脚本前，执行自定义容器实现时失败（Executing the custom container implementation failed），提示联系自托管运行器管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597180

- **stage-b-test-4-npu-a3**: 日志显示在加载模型权重时出现错误："Executing the custom container implementation failed"，建议检查自托管运行器容器配置及NPU资源状态。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597190

- **stage-b-test-1-npu-a2 (0)**: 日志显示在运行测试脚本后，出现错误：Executing the custom container implementation failed，提示联系自托管运行器管理员，表明是CI基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597207

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含环境准备和清理步骤，缺少测试运行阶段的输出，无法判断失败原因。可能因日志截断或作业在测试执行前已失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30522699084/job/90806597579


## [Run #30522606711](https://github.com/sgl-project/sglang/actions/runs/30522606711)
- **分支**: `feat/spectrum`
- **总耗时**: 36.7min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30522606711

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30522606711/job/90806389591) |
| multimodal-gen-test-2-npu-a3 | 34.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30522606711/job/90806389646) |


## [Run #30521289273](https://github.com/sgl-project/sglang/actions/runs/30521289273)
- **分支**: `add_mxfp4w4a8_quantization_for_npu`
- **总耗时**: 57.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30521289273

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141308) |
| stage-b-test-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141320) |
| stage-b-test-1-npu-a2 (1) | 32.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141358) |
| stage-b-test-1-npu-a2 (0) | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141362) |
| stage-b-test-16-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141393) |
| stage-b-test-2-npu-a2 (0) | 16.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141404) |
| multimodal-gen-test-1-npu-a3 | 33.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141407) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141426) |
| multimodal-gen-test-2-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141448) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521289273/job/90802141909) |


## [Run #30521287682](https://github.com/sgl-project/sglang/actions/runs/30521287682)
- **分支**: `cp/interleave-v2`
- **总耗时**: 42.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30521287682

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109210) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109222) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109223) |
| stage-b-test-1-npu-a2 (0) | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109228) |
| multimodal-gen-test-1-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109233) |
| stage-b-test-1-npu-a2 (1) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109261) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109271) |
| multimodal-gen-test-2-npu-a3 | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109276) |
| stage-b-test-16-npu-a3 | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109334) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521287682/job/90802109660) |


## [Run #30521283647](https://github.com/sgl-project/sglang/actions/runs/30521283647)
- **分支**: `mick/encoder-parallel-unified`
- **总耗时**: 42.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30521283647

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 41.9min | 其他 | 日志未显示测试失败，仅包含Node.js版本弃用警告和工件上传提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30521283647/job/90802122251) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败或错误信息，仅包含Node.js 20弃用警告和'No files were found'的工件上传提示，作业可能因其他原因被标记为失败，但日志不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/30521283647/job/90802122251

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30521283647/job/90802122215) |


## [Run #30520684820](https://github.com/sgl-project/sglang/actions/runs/30520684820)
- **分支**: `feat/add-cosmos3-edge-distil`
- **总耗时**: 49.6min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30520684820

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 43.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520684820/job/90800254619) |
| multimodal-gen-test-1-npu-a3 | 48.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520684820/job/90800254676) |


## [Run #30520612956](https://github.com/sgl-project/sglang/actions/runs/30520612956)
- **分支**: `feat/spectrum`
- **总耗时**: 35.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30520612956

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 33.7min | 其他 | 作业日志不完整，未显示测试执行和失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520612956/job/90800050343) |
| multimodal-gen-test-2-npu-a3 | 33.7min | 其他 | 日志未显示测试失败的具体错误，仅包含Node.js版本弃用警告和工件上传提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520612956/job/90800050362) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含环境准备和清理步骤，缺少测试运行、断言失败或错误堆栈等关键信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520612956/job/90800050343

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的关键信息，仅记录了Node.js 20弃用警告、工件上传时未找到文件等非致命问题，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520612956/job/90800050362


## [Run #30520598726](https://github.com/sgl-project/sglang/actions/runs/30520598726)
- **分支**: `dspark-integration`
- **总耗时**: 46.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30520598726

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912876) |
| stage-b-test-16-npu-a3 | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912906) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912929) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912932) |
| stage-b-test-1-npu-a2 (1) | 32.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912933) |
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912939) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912946) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912970) |
| multimodal-gen-test-2-npu-a3 | 46.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799912982) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520598726/job/90799913417) |


## [Run #30520447374](https://github.com/sgl-project/sglang/actions/runs/30520447374)
- **分支**: `main`
- **总耗时**: 33.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30520447374

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 29.2min | 其他 | 作业日志不完整，未显示测试执行与失败信息，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989927) |
| multimodal-gen-test-1-npu-a3 | 28.9min | 其他 | 日志未显示测试失败的具体原因，仅包含环境警告和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989936) |
| stage-b-test-4-npu-a3 | 26.8min | 环境问题 | 自定义容器执行失败，可能是容器环境或配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989943) |
| stage-b-test-1-npu-a2 (1) | 29.1min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989946) |
| stage-b-test-1-npu-a2 (0) | 29.2min | 环境问题 | 自定义容器执行失败，可能是容器或运行环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989952) |

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备、Node.js 版本警告及清理步骤，缺少测试运行、断言失败或错误堆栈等关键信息，可能因日志截断或作业未实际运行测试而失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989927

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的关键信息，仅有Node.js版本弃用警告和工件上传提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989936

- **stage-b-test-4-npu-a3**: 日志显示 'Executing the custom container implementation failed'，提示联系自托管运行器管理员，表明容器执行环境异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989943

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行到约75%时，自定义容器执行失败（Executing the custom container implementation failed），同时有Node.js版本弃用警告，但核心原因是容器环境异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989946

- **stage-b-test-1-npu-a2 (0)**: 日志显示模型加载成功，但随后出现 'Executing the custom container implementation failed' 错误，提示联系自托管运行器管理员，表明容器执行环境存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989952

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989930) |
| stage-b-test-16-npu-a3 | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989937) |
| stage-b-test-2-npu-a2 (0) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989939) |
| stage-b-test-2-npu-a2 (1) | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799989987) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30520447374/job/90799990329) |


## [Run #30520322845](https://github.com/sgl-project/sglang/actions/runs/30520322845)
- **分支**: `feat/spectrum`
- **总耗时**: 5.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30520322845

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.2min | 其他 | 日志中未显示测试执行失败，仅包含环境警告和工件上传信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520322845/job/90799076362) |
| multimodal-gen-test-2-npu-a3 | 5.2min | 其他 | 作业日志不完整，未显示测试执行和失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520322845/job/90799076377) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含Node.js版本弃用警告和工件上传步骤（未找到文件），未显示实际测试执行或失败信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520322845/job/90799076362

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备和清理步骤，缺少实际测试运行及失败原因，无法判断具体失败类型。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520322845/job/90799076377


## [Run #30520291390](https://github.com/sgl-project/sglang/actions/runs/30520291390)
- **分支**: `main`
- **总耗时**: 6.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30520291390

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 2.3min | 环境问题 | 自定义容器执行失败，可能是torch_npu下载或安装出错。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052108) |
| stage-b-test-16-npu-a3 | 2.3min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052111) |
| multimodal-gen-test-1-npu-a3 | 5.1min | 其他 | 日志未显示测试失败原因，仅包含Node.js版本弃用警告和工件上传信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052119) |
| stage-b-test-1-npu-a2 (1) | 2.3min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052124) |
| multimodal-gen-test-2-npu-a3 | 2.3min | 其他 | 日志未显示测试失败原因，仅包含Node.js版本弃用警告和工件上传信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052141) |
| stage-a-unit-test-npu | 2.2min | 环境问题 | 自定义容器执行失败，导致作业中止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052143) |
| stage-b-test-4-npu-a3 | 2.5min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052147) |
| stage-b-test-2-npu-a2 (0) | 2.2min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052161) |
| stage-b-test-2-npu-a2 (1) | 2.2min | 环境问题 | 安装triton-ascend包时失败，自定义容器执行出错。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052171) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 2.5min | 其他 | 作业日志不完整，未显示测试执行和失败信息，仅包含环境准备和清理步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052564) |

- **stage-b-test-1-npu-a2 (0)**: 在安装torch_npu-2.10.0时，下载完成后执行容器实现失败，提示联系自托管运行器管理员，属于环境配置或依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052108

- **stage-b-test-16-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，建议联系自托管运行器管理员。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052111

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅记录了Node.js 20弃用警告和上传工件时未找到文件。需要查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052119

- **stage-b-test-1-npu-a2 (1)**: 在安装torch-npu时，自定义容器实现执行失败，提示联系自托管运行程序管理员，可能是容器环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052124

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示Node.js 20弃用警告和上传工件时未找到文件。需要查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052141

- **stage-a-unit-test-npu**: 日志显示在安装torch-npu成功后，出现错误：Executing the custom container implementation failed，提示联系自托管运行器管理员，属于CI环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052143

- **stage-b-test-4-npu-a3**: 在安装依赖包后，自定义容器实现执行失败，提示联系自托管运行器管理员，可能是容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052147

- **stage-b-test-2-npu-a2 (0)**: 作业在安装torch-npu后，执行自定义容器实现时失败，提示请联系自托管运行器管理员，可能是容器环境配置或权限问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052161

- **stage-b-test-2-npu-a2 (1)**: 在安装triton-ascend==3.2.1.dev20260530时，pip下载失败，导致自定义容器实现执行错误，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052171

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含GitHub Actions初始化、Node.js版本警告及清理操作，未输出测试运行、性能指标或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30520291390/job/90799052564


## [Run #30518909754](https://github.com/sgl-project/sglang/actions/runs/30518909754)
- **分支**: `main`
- **总耗时**: 26.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30518909754

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.8min | 其他 | 作业日志不完整，未显示测试执行与失败信息，仅包含环境准备和清理步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910882) |
| multimodal-gen-test-2-npu-a3 | 25.8min | 其他 | 作业日志不完整，未显示测试执行与失败信息，仅包含环境准备和Node版本弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910908) |
| stage-b-test-1-npu-a2 (0) | 25.8min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910927) |
| stage-b-test-1-npu-a2 (1) | 25.8min | 环境问题 | 自定义容器执行失败，可能是容器或运行环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910929) |
| stage-b-test-4-npu-a3 | 25.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910943) |

- **multimodal-gen-test-1-npu-a3**: 日志中缺少测试运行、断言失败或错误堆栈等关键内容，无法判断失败原因。可能因作业被提前中止或日志截断导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910882

- **multimodal-gen-test-2-npu-a3**: 日志截断于作业早期阶段，缺少测试运行、断言失败或错误堆栈等关键信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910908

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重后，执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，可能是容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910927

- **stage-b-test-1-npu-a2 (1)**: 日志显示“Executing the custom container implementation failed”，且伴随Node.js版本弃用警告，表明自托管运行器环境配置问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910929

- **stage-b-test-4-npu-a3**: 日志显示在加载模型权重时出现 `##[error]Executing the custom container implementation failed`，提示联系自托管运行器管理员，表明容器环境异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910943

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910910) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910912) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910917) |
| stage-b-test-16-npu-a3 | 18.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794910934) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518909754/job/90794911213) |


## [Run #30518603273](https://github.com/sgl-project/sglang/actions/runs/30518603273)
- **分支**: `fix-prefill-delayer-slot-delay-bound`
- **总耗时**: 43.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30518603273

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024633) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024643) |
| multimodal-gen-test-2-npu-a3 | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024657) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024678) |
| stage-b-test-4-npu-a3 | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024686) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024692) |
| stage-b-test-1-npu-a2 (1) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024698) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024707) |
| multimodal-gen-test-1-npu-a3 | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794024714) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30518603273/job/90794025065) |


## [Run #30517904810](https://github.com/sgl-project/sglang/actions/runs/30517904810)
- **分支**: `tom/revert-pr10414`
- **总耗时**: 44.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30517904810

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955175) |
| multimodal-gen-test-1-npu-a3 | 38.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955176) |
| stage-b-test-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955183) |
| multimodal-gen-test-2-npu-a3 | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955195) |
| stage-b-test-1-npu-a2 (0) | 43.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955205) |
| stage-b-test-1-npu-a2 (1) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955221) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955235) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955240) |
| stage-b-test-16-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955245) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517904810/job/90791955547) |


## [Run #30517742933](https://github.com/sgl-project/sglang/actions/runs/30517742933)
- **分支**: `fp8_kv_cache_rebase`
- **总耗时**: 45.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30517742933

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 34.5min | 代码错误 | 测试用例 test_npu_hicache_mla.py 执行失败，返回非零退出码。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297278) |
| stage-b-test-1-npu-a2 (0) | 34.6min | 代码错误 | 测试用例 test_npu_autoround_moe.py 执行失败，返回非零退出码。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297311) |

- **stage-b-test-4-npu-a3**: 在 HiCache 测试中，test_npu_hicache_mla.py 失败（exit code 1），其他3个测试通过。该测试耗时428秒，可能涉及 MLA 相关功能异常或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297278

- **stage-b-test-1-npu-a2 (0)**: 在 stage-b-test-1-npu-a2 作业中，5个测试用例有3个通过，但 test_npu_autoround_moe.py 失败（exit code 1），导致整体作业失败。具体错误原因需查看该测试日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297311

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297280) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297299) |
| stage-b-test-1-npu-a2 (1) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297302) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297303) |
| multimodal-gen-test-1-npu-a3 | 35.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297307) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297332) |
| multimodal-gen-test-2-npu-a3 | 44.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297340) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30517742933/job/90791297512) |


## [Run #30516414514](https://github.com/sgl-project/sglang/actions/runs/30516414514)
- **分支**: `ds_v4_xpu_fused_q_norm_rope`
- **总耗时**: 42.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30516414514

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30516414514/job/90787251655) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30516414514/job/90787251679) |
| stage-b-test-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30516414514/job/90787251680) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30516414514/job/90787251688) |
| stage-b-test-2-npu-a2 (1) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30516414514/job/90787251693) |
| stage-b-test-1-npu-a2 (1) | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30516414514/job/90787251699) |
| stage-b-test-16-npu-a3 | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30516414514/job/90787251727) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30516414514/job/90787252059) |


## [Run #30515575737](https://github.com/sgl-project/sglang/actions/runs/30515575737)
- **分支**: `fix-teardown-gpu-idle`
- **总耗时**: 42.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30515575737

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515575737/job/90784575919) |
| stage-b-test-4-npu-a3 | 37.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515575737/job/90784575922) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515575737/job/90784575925) |
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515575737/job/90784575943) |
| stage-b-test-1-npu-a2 (0) | 41.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515575737/job/90784575952) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515575737/job/90784575965) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515575737/job/90784575970) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515575737/job/90784576369) |


## [Run #30515206493](https://github.com/sgl-project/sglang/actions/runs/30515206493)
- **分支**: `dit-full-forward-cuda-graph`
- **总耗时**: 41.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30515206493

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515206493/job/90783461017) |
| multimodal-gen-test-1-npu-a3 | 28.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30515206493/job/90783461024) |


## [Run #30513475922](https://github.com/sgl-project/sglang/actions/runs/30513475922)
- **分支**: `main`
- **总耗时**: 67.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30513475922

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 37.8min | 其他 | 作业日志不完整，未显示测试执行与失败信息 | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269131) |

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备、Node版本警告和清理步骤，缺少实际测试运行及失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269131

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269087) |
| multimodal-gen-test-1-npu-a3 | 44.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269092) |
| stage-b-test-1-npu-a2 (0) | 42.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269105) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269116) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269122) |
| stage-b-test-16-npu-a3 | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269134) |
| stage-b-test-4-npu-a3 | 41.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269138) |
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269145) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513475922/job/90778269312) |


## [Run #30513243253](https://github.com/sgl-project/sglang/actions/runs/30513243253)
- **分支**: `lsyin/draft-extend-device-timer`
- **总耗时**: 57.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30513243253

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333749) |
| stage-b-test-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333763) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333764) |
| stage-b-test-1-npu-a2 (0) | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333774) |
| multimodal-gen-test-1-npu-a3 | 36.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333776) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333824) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333854) |
| stage-b-test-2-npu-a2 (0) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333888) |
| multimodal-gen-test-2-npu-a3 | 51.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780333893) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513243253/job/90780334129) |


## [Run #30513209183](https://github.com/sgl-project/sglang/actions/runs/30513209183)
- **分支**: `dit-full-forward-cuda-graph`
- **总耗时**: 44.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30513209183

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 32.1min | 其他 | 作业日志不完整，未显示测试失败或错误信息，仅包含环境警告和上传工件提示无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30513209183/job/90777511561) |

- **multimodal-gen-test-2-npu-a3**: 日志截断，仅包含Node.js版本弃用警告和上传工件时未找到失败文件，未提供实际测试失败原因，无法判断具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/30513209183/job/90777511561

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30513209183/job/90777511533) |


## [Run #30511903357](https://github.com/sgl-project/sglang/actions/runs/30511903357)
- **分支**: `dev/fanshuaishuai/feat_overlap_image_load`
- **总耗时**: 71.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30511903357

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 4.5min | 环境问题 | pip下载依赖时网络连接中断导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623681) |

- **stage-a-unit-test-npu**: 在安装Python依赖过程中，pip下载文件时出现IncompleteRead错误，网络连接中断导致下载不完整，最终退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623681

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623650) |
| stage-b-test-1-npu-a2 (1) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623668) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623669) |
| stage-b-test-1-npu-a2 (0) | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623672) |
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623678) |
| multimodal-gen-test-2-npu-a3 | 33.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623680) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623686) |
| stage-b-test-16-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773623695) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511903357/job/90773624069) |


## [Run #30511790651](https://github.com/sgl-project/sglang/actions/runs/30511790651)
- **分支**: `fix-teardown-gpu-idle`
- **总耗时**: 63.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30511790651

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511790651/job/90773346823) |
| stage-b-test-1-npu-a2 (1) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511790651/job/90773346828) |
| stage-b-test-1-npu-a2 (0) | 43.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511790651/job/90773346832) |
| stage-b-test-16-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511790651/job/90773346835) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511790651/job/90773346837) |
| stage-b-test-2-npu-a2 (1) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511790651/job/90773346847) |
| stage-b-test-4-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511790651/job/90773346856) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511790651/job/90773347008) |


## [Run #30511535181](https://github.com/sgl-project/sglang/actions/runs/30511535181)
- **分支**: `feat/lingbot-video-moe-30b`
- **总耗时**: 79.2min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30511535181

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511535181/job/90772552484) |
| multimodal-gen-test-2-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511535181/job/90772552493) |


## [Run #30511455052](https://github.com/sgl-project/sglang/actions/runs/30511455052)
- **分支**: `fix-teardown-gpu-idle`
- **总耗时**: 8.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30511455052

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318196) |
| stage-b-test-16-npu-a3 | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318200) |
| stage-b-test-1-npu-a2 (0) | 7.5min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318218) |
| stage-b-test-1-npu-a2 (1) | 7.3min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318235) |
| stage-b-test-2-npu-a2 (0) | 7.5min | 环境问题 | 自定义容器执行失败，可能是Kubernetes Pod异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318253) |
| stage-b-test-2-npu-a2 (1) | 7.3min | 环境问题 | 自定义容器执行失败，健康检查返回503 | [job link](https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318266) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.2min | 环境问题 | 依赖的 blob 文件不存在导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318520) |

- **stage-b-test-4-npu-a3**: 作业尝试访问 Azure Blob 存储中的某个 blob，但该 blob 已被删除或路径错误，返回 BlobNotFound 错误。这属于外部依赖或配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318196

- **stage-b-test-16-npu-a3**: 作业尝试访问 Azure Blob 存储中的某个 blob，但该 blob 已被删除或路径错误，返回 BlobNotFound 错误。这属于外部依赖或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318200

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行中突然出现"Executing the custom container implementation failed"错误，提示联系自托管运行器管理员，表明是NPU容器环境异常导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318218

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行中突然出现'Executing the custom container implementation failed'错误，提示联系自托管运行器管理员，表明是NPU容器环境异常导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318235

- **stage-b-test-2-npu-a2 (0)**: 日志显示'Executing the custom container implementation failed'，且作业在运行测试时突然中断，未完成测试，表明容器环境或K8s调度出现问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318253

- **stage-b-test-2-npu-a2 (1)**: 服务启动后，/health_generate 接口持续返回503 Service Unavailable，导致容器执行失败，可能是模型加载或初始化异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318266

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 Azure Blob 存储中找不到指定的 blob 文件（BlobNotFound），可能是模型权重或数据文件被删除、路径错误或未上传，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318520

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30511455052/job/90772318187) |


---
*Auto-generated by npu_pr_monitor.py*