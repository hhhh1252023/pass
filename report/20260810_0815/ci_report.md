# NPU CI 执行监控
**生成时间**: 2026-08-10 00:15 UTC
**分析 Run 数**: 26

---

## [Run #30027312368](https://github.com/sgl-project/sglang/actions/runs/30027312368)
- **分支**: `jthomson04/kv-event-coalesce`
- **总耗时**: 13.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30027312368

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 12.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275105980) |
| multimodal-gen-test-1-npu-a3 | 10.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106023) |
| stage-b-test-2-npu-a2 (0) | 12.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106035) |
| multimodal-gen-test-2-npu-a3 | 9.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106040) |
| stage-b-test-1-npu-a2 (1) | 9.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106047) |
| stage-b-test-2-npu-a2 (1) | 10.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106049) |
| stage-b-test-16-npu-a3 | 9.6min | 环境问题 | NPU 容器执行失败，模型权重加载卡住导致 watchdog 超时。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106054) |
| stage-b-test-1-npu-a2 (0) | 6.7min | 环境问题 | 自定义容器实现执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106147) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 12.2min | 其他 | 日志被截断，未显示测试执行结果，仅见作业清理和Node.js弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106602) |

- **stage-b-test-4-npu-a3**: 日志显示服务启动正常，但在获取环境变量后，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275105980

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。可能因日志截断或作业在测试前被取消，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106023

- **stage-b-test-2-npu-a2 (0)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106035

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node 20弃用警告和上传artifact时未找到文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106040

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行到约6%时，自定义容器实现执行失败，提示联系自托管runner管理员，可能因NPU资源或容器环境不稳定导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106047

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常，但中途出现“Executing the custom container implementation failed”错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106049

- **stage-b-test-16-npu-a3**: 日志显示在加载 MoE 模型权重时出现 libtorch_python.so 相关错误，随后 scheduler watchdog 超时，最终自定义容器执行失败，疑似 NPU 环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106054

- **stage-b-test-1-npu-a2 (0)**: 日志显示服务启动后健康检查返回503，随后出现NPU算子回退警告，最终自定义容器执行失败，可能是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106147

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分被省略，无法定位具体失败原因。仅看到作业结束时的清理步骤和Node.js 20弃用警告，未出现测试失败或错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89275106602

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| pr-gate / pr-gate | 0.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30027312368/job/89274971042) |


## [Run #29992866331](https://github.com/sgl-project/sglang/actions/runs/29992866331)
- **分支**: `fix/rope-config-and-vl-weight-loading`
- **总耗时**: 6.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29992866331

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 3.3min | 环境问题 | 下载triton-ascend依赖时自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159623919) |
| multimodal-gen-test-1-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159623946) |
| stage-b-test-4-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159623954) |
| stage-b-test-1-npu-a2 (1) | 3.7min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159623969) |
| stage-b-test-16-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624009) |
| multimodal-gen-test-2-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624011) |
| stage-b-test-1-npu-a2 (0) | 3.7min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624070) |
| stage-b-test-2-npu-a2 (1) | 1.9min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624072) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624856) |

- **stage-b-test-2-npu-a2 (0)**: 在安装triton-ascend==3.2.1.dev20260530（188.5MB）过程中，自定义容器实现执行失败，可能是网络或容器资源问题，非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159623919

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159623946

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159623954

- **stage-b-test-1-npu-a2 (1)**: 日志显示在apt更新过程中出现网络获取错误（jammy-updates InRelease被忽略），随后runner报错“Executing the custom container implementation failed”，属于基础设施或网络环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159623969

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置变更导致，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624009

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624011

- **stage-b-test-1-npu-a2 (0)**: 日志显示在CMake配置阶段，runner报错“Executing the custom container implementation failed”，属于runner环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624070

- **stage-b-test-2-npu-a2 (1)**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624072

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明CI作业尝试访问的远程存储资源缺失，可能是日志上传或下载路径配置错误，或资源被清理，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29992866331/job/89159624856


## [Run #29989043924](https://github.com/sgl-project/sglang/actions/runs/29989043924)
- **分支**: `main`
- **总耗时**: 46.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29989043924

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 2.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426438) |
| multimodal-gen-test-2-npu-a3 | 8.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426473) |
| multimodal-gen-test-1-npu-a3 | 8.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426477) |
| stage-b-test-4-npu-a3 | 8.0min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426543) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.6min | 环境问题 | 作业在启动后立即失败，未执行实际测试，可能因运行环境或资源问题导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426860) |

- **stage-b-test-16-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426438

- **multimodal-gen-test-2-npu-a3**: 日志截断，缺少核心测试执行部分。仅看到上传diffusion-failures工件时提示无文件，说明测试可能未产生失败样本，但无法确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426473

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426477

- **stage-b-test-4-npu-a3**: 日志显示在测试运行过程中，runner报告“Executing the custom container implementation failed”，提示联系管理员，属于runner环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426543

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试脚本，且无错误信息，可能因runner环境异常或资源分配失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426860

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426492) |
| stage-b-test-1-npu-a2 (0) | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426494) |
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426499) |
| stage-b-test-2-npu-a2 (1) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29989043924/job/89147426652) |


## [Run #29988777660](https://github.com/sgl-project/sglang/actions/runs/29988777660)
- **分支**: `xyf/decode_slice`
- **总耗时**: 74.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29988777660

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146617858) |
| stage-b-test-4-npu-a3 | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146617863) |
| multimodal-gen-test-1-npu-a3 | 35.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146617881) |
| stage-b-test-16-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146617886) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146617902) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146617920) |
| stage-b-test-2-npu-a2 (1) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146617962) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146617969) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29988777660/job/89146618275) |


## [Run #29987940620](https://github.com/sgl-project/sglang/actions/runs/29987940620)
- **分支**: `amd/enable-mamba-transfer-kernel-rocm`
- **总耗时**: 85.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29987940620

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 3.5min | 环境问题 | pip 下载依赖时网络连接中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975488) |

- **stage-b-test-1-npu-a2 (1)**: 在安装 Python 依赖过程中，pip 从远程源下载包时发生 IncompleteRead 错误，网络连接中断，导致进程退出码 1，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975488

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975440) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975473) |
| multimodal-gen-test-2-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975478) |
| multimodal-gen-test-1-npu-a3 | 32.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975485) |
| stage-b-test-2-npu-a2 (1) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975490) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975531) |
| stage-b-test-2-npu-a2 (0) | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975557) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987940620/job/89143975806) |


## [Run #29987595560](https://github.com/sgl-project/sglang/actions/runs/29987595560)
- **分支**: `fix/glm-tool-parser-escapes`
- **总耗时**: 90.1min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29987595560

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142907690) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142907695) |
| multimodal-gen-test-1-npu-a3 | 31.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142907709) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142907727) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142907765) |
| multimodal-gen-test-2-npu-a3 | 36.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142907768) |
| stage-b-test-4-npu-a3 | 40.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142907817) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142907856) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29987595560/job/89142908168) |


## [Run #29986820007](https://github.com/sgl-project/sglang/actions/runs/29986820007)
- **分支**: `fix/hisparse-host-backed-max-request-length`
- **总耗时**: 78.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29986820007

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 34.6min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503702) |
| stage-b-test-1-npu-a2 (1) | 3.1min | 环境问题 | pip安装依赖时网络连接中断，导致下载不完整。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503727) |

- **multimodal-gen-test-2-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503702

- **stage-b-test-1-npu-a2 (1)**: 在安装Python包时，pip从远程下载文件过程中出现IncompleteRead错误，连接中断，实际读取118MB但预期还有70MB未下载，导致安装失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503727

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503669) |
| stage-b-test-4-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503686) |
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503689) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503690) |
| multimodal-gen-test-1-npu-a3 | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503695) |
| stage-b-test-1-npu-a2 (0) | 44.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140503733) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29986820007/job/89140504140) |


## [Run #29985872214](https://github.com/sgl-project/sglang/actions/runs/29985872214)
- **分支**: `py/fix-mooncake-tp-2`
- **总耗时**: 84.6min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29985872214

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169253) |
| stage-b-test-4-npu-a3 | 38.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169315) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169316) |
| stage-b-test-16-npu-a3 | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169318) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169344) |
| multimodal-gen-test-2-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169347) |
| stage-b-test-1-npu-a2 (0) | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169389) |
| stage-b-test-1-npu-a2 (1) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169423) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29985872214/job/89141169788) |


## [Run #29984796109](https://github.com/sgl-project/sglang/actions/runs/29984796109)
- **分支**: `fix_aiter_preshuffle_mqa`
- **总耗时**: 57.2min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29984796109

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256202) |
| stage-b-test-16-npu-a3 | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256203) |
| stage-b-test-2-npu-a2 (0) | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256219) |
| multimodal-gen-test-2-npu-a3 | 43.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256227) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256230) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256236) |
| multimodal-gen-test-1-npu-a3 | 35.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256239) |
| stage-b-test-1-npu-a2 (1) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256286) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984796109/job/89134256431) |


## [Run #29984616875](https://github.com/sgl-project/sglang/actions/runs/29984616875)
- **分支**: `amx_sgl-diffusion_opt`
- **总耗时**: 57.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29984616875

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133713690) |
| multimodal-gen-test-1-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133713696) |
| multimodal-gen-test-2-npu-a3 | 34.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133713710) |
| stage-b-test-1-npu-a2 (1) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133713714) |
| stage-b-test-4-npu-a3 | 40.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133713725) |
| stage-b-test-1-npu-a2 (0) | 43.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133713729) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133713744) |
| stage-b-test-2-npu-a2 (0) | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133713760) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29984616875/job/89133714021) |


## [Run #29981491500](https://github.com/sgl-project/sglang/actions/runs/29981491500)
- **分支**: `lsyin/spec-retract-order`
- **总耗时**: 8.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29981491500

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086418) |
| stage-b-test-2-npu-a2 (0) | 7.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086424) |
| stage-b-test-2-npu-a2 (1) | 7.1min | 环境问题 | NPU 服务启动后健康检查持续返回 503，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086430) |
| stage-b-test-16-npu-a3 | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086431) |
| multimodal-gen-test-1-npu-a3 | 1.6min | 环境问题 | 作业因缺少diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086437) |
| multimodal-gen-test-2-npu-a3 | 5.4min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，可能因前置步骤失败或测试未运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086443) |
| stage-b-test-1-npu-a2 (0) | 7.5min | 环境问题 | NPU服务启动后健康检查返回503，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086455) |
| stage-b-test-1-npu-a2 (1) | 7.4min | 环境问题 | 自定义容器执行失败，测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086457) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.9min | 环境问题 | 测试未生成metrics.json，导致性能测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086707) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086418

- **stage-b-test-2-npu-a2 (0)**: 日志显示在启动Qwen2.5-7B-Instruct模型服务时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086424

- **stage-b-test-2-npu-a2 (1)**: 日志显示服务启动后 /health_generate 接口多次返回 503 Service Unavailable，说明模型未就绪或推理进程异常，最终容器执行失败。可能是 NPU 资源或模型加载问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086430

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086431

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤未找到diffusion-failures/文件，说明测试未生成失败产物，可能因环境配置或前置步骤失败导致作业未正常运行，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086437

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，且提示未找到diffusion-failures目录，说明测试未产生失败文件，可能因环境问题或测试未执行导致作业提前结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086443

- **stage-b-test-1-npu-a2 (0)**: 服务启动后/health_generate接口持续返回503，说明模型未就绪或NPU资源异常，最终容器实现执行失败，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086455

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行到TestAscendSamplingBackend.test_mmlu时，自定义容器实现执行失败，导致作业提前终止，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086457

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业在运行性能测试后未找到/tmp/metrics.json文件，无法上传性能指标，可能因测试执行异常或环境配置问题导致性能数据未生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/29981491500/job/89124086707


## [Run #29980993470](https://github.com/sgl-project/sglang/actions/runs/29980993470)
- **分支**: `main`
- **总耗时**: 46.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29980993470

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980993470/job/89122569870) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980993470/job/89122569889) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980993470/job/89122569898) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980993470/job/89122569901) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980993470/job/89122569904) |
| stage-b-test-4-npu-a3 | 41.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980993470/job/89122569921) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980993470/job/89122570121) |


## [Run #29980197452](https://github.com/sgl-project/sglang/actions/runs/29980197452)
- **分支**: `enable_xpu_platform_support`
- **总耗时**: 57.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29980197452

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263317) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263333) |
| stage-b-test-4-npu-a3 | 40.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263336) |
| stage-b-test-1-npu-a2 (1) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263340) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263341) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263351) |
| multimodal-gen-test-1-npu-a3 | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263352) |
| multimodal-gen-test-2-npu-a3 | 54.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263358) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29980197452/job/89120263517) |


## [Run #29979964282](https://github.com/sgl-project/sglang/actions/runs/29979964282)
- **分支**: `kimi-deterministic/3-deepep-normal-bf16-deepgemm`
- **总耗时**: 42.7min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29979964282

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579291) |
| multimodal-gen-test-2-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579312) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579332) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579338) |
| stage-b-test-16-npu-a3 | 19.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579339) |
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579344) |
| stage-b-test-4-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579348) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579354) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979964282/job/89119579637) |


## [Run #29979279003](https://github.com/sgl-project/sglang/actions/runs/29979279003)
- **分支**: `idhanani/dyn-29465-mm-inputs-msgpack`
- **总耗时**: 51.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29979279003

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117602988) |
| multimodal-gen-test-2-npu-a3 | 34.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117602989) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117602996) |
| stage-b-test-16-npu-a3 | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117603000) |
| multimodal-gen-test-1-npu-a3 | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117603007) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117603037) |
| stage-b-test-2-npu-a2 (0) | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117603038) |
| stage-b-test-1-npu-a2 (0) | 43.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117603103) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29979279003/job/89117603234) |


## [Run #29976391198](https://github.com/sgl-project/sglang/actions/runs/29976391198)
- **分支**: `libinta/xpu_lmcache`
- **总耗时**: 243.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29976391198

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 241.4min | 环境问题 | 测试环境缺少Python依赖包tabulate，导致run_suite.py无法导入模块而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995668) |

- **stage-b-test-1-npu-a2 (1)**: 在stage-b-test-1-npu-a2作业中，执行test/run_suite.py时因缺少tabulate模块报ModuleNotFoundError，属于CI环境依赖未安装完整，非代码逻辑问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995668

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995631) |
| multimodal-gen-test-2-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995651) |
| stage-b-test-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995662) |
| stage-b-test-16-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995663) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995676) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995682) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995702) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29976391198/job/89108995993) |


## [Run #29972558923](https://github.com/sgl-project/sglang/actions/runs/29972558923)
- **分支**: `pr_branch_shequ`
- **总耗时**: 44.7min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29972558923

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29972558923/job/89119600672) |
| stage-b-test-16-npu-a3 | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29972558923/job/89119600675) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29972558923/job/89119600689) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29972558923/job/89119600697) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29972558923/job/89119600723) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29972558923/job/89119600727) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29972558923/job/89119600845) |


## [Run #29943521104](https://github.com/sgl-project/sglang/actions/runs/29943521104)
- **分支**: `jeeja/add_xpu_support_disaggregate`
- **总耗时**: 21.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29943521104

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181154) |
| stage-b-test-1-npu-a2 (1) | 20.2min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181179) |
| stage-b-test-2-npu-a2 (1) | 18.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181185) |
| stage-b-test-16-npu-a3 | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181190) |
| stage-b-test-4-npu-a3 | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181204) |
| multimodal-gen-test-2-npu-a3 | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181227) |
| stage-b-test-1-npu-a2 (0) | 19.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181230) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181720) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181154

- **stage-b-test-1-npu-a2 (1)**: 日志显示在加载模型权重时（Multi-thread loading shards 0%）出现错误，提示自定义容器实现执行失败，可能是NPU环境或容器配置问题导致进程异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181179

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常，但在执行过程中出现错误："Executing the custom container implementation failed. Please contact your self hosted runner administrator."，表明自托管运行器的容器环境出现问题，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181185

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181190

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181204

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181227

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但在执行第二个测试时，自定义容器实现失败，提示联系自托管runner管理员，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181230

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181720

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29943521104/job/89003181184) |


## [Run #29942389900](https://github.com/sgl-project/sglang/actions/runs/29942389900)
- **分支**: `pranjalssh/rope-mixed-q-dtype`
- **总耗时**: 7.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29942389900

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445886) |
| stage-b-test-4-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445902) |
| multimodal-gen-test-1-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445903) |
| multimodal-gen-test-2-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445915) |
| stage-b-test-1-npu-a2 (1) | 6.6min | 环境问题 | NPU服务启动后健康检查返回503，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445928) |
| stage-b-test-1-npu-a2 (0) | 6.9min | 环境问题 | NPU后端算子不支持导致服务启动后健康检查失败，最终容器执行异常退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445930) |
| stage-b-test-2-npu-a2 (0) | 6.9min | 环境问题 | NPU健康检查返回503，服务未就绪导致容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445933) |
| stage-b-test-2-npu-a2 (1) | 6.8min | 环境问题 | NPU 服务健康检查失败导致作业中止 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006446044) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，属于外部资源缺失。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006446670) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445886

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445902

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445903

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或模型权重在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445915

- **stage-b-test-1-npu-a2 (1)**: 服务在18:03:24启动成功，但后续/health_generate接口持续返回503，表明模型未就绪或推理异常，最终容器实现执行失败，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445928

- **stage-b-test-1-npu-a2 (0)**: 日志显示服务启动后/health_generate返回503，且存在aten::_assert_async算子不支持NPU回退CPU的警告，导致预填充吞吐极低（3.39 token/s），最终自定义容器执行失败，属于NPU环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445930

- **stage-b-test-2-npu-a2 (0)**: 日志显示服务启动后/health_generate接口连续返回503，说明模型未完成加载或NPU资源异常，最终自定义容器实现失败，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006445933

- **stage-b-test-2-npu-a2 (1)**: 服务启动后 /health_generate 返回 503，随后自定义容器执行失败，可能是 NPU 资源或环境配置异常，导致服务未就绪。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006446044

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业失败原因是下载或访问Azure Blob存储中的文件时返回BlobNotFound错误，可能是日志文件或依赖资源被删除、路径错误或存储配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/29942389900/job/89006446670


## [Run #29941352053](https://github.com/sgl-project/sglang/actions/runs/29941352053)
- **分支**: `mmangkad/use-pip-fa4-by-default`
- **总耗时**: 51.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29941352053

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 12.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872404) |
| stage-b-test-16-npu-a3 | 10.9min | 超时 | 模型加载shards时Scheduler watchdog超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872443) |
| multimodal-gen-test-1-npu-a3 | 15.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872497) |
| stage-b-test-4-npu-a3 | 14.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872567) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.0min | 其他 | 日志被截断，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995873043) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误或失败信息，仅显示Node.js版本弃用警告和上传artifact时无文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872404

- **stage-b-test-16-npu-a3**: 日志显示在加载模型分片（shards）过程中，Scheduler watchdog超时（300秒），同时加载进度缓慢（76%时耗时已超4分钟），最终触发自定义容器执行失败，作业中止。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872443

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但实际失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872497

- **stage-b-test-4-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872567

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志仅包含作业启动和清理阶段，未显示测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995873043

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872381) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872440) |
| stage-b-test-1-npu-a2 (1) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872458) |
| stage-b-test-1-npu-a2 (0) | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29941352053/job/88995872491) |


## [Run #29941187270](https://github.com/sgl-project/sglang/actions/runs/29941187270)
- **分支**: `online-nvfp4-to-mxfp4-convert`
- **总耗时**: 54.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29941187270

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281408) |
| multimodal-gen-test-2-npu-a3 | 15.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281451) |
| stage-b-test-4-npu-a3 | 15.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281455) |
| stage-b-test-16-npu-a3 | 16.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281485) |
| stage-b-test-1-npu-a2 (0) | 33.5min | 代码错误 | NPU测试中test_npu_autoround_moe.py失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281523) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281408

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有GitHub Actions环境准备、Node版本警告及上传artifact时未找到diffusion-failures目录的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281451

- **stage-b-test-4-npu-a3**: 日志显示在捕获批次过程中，自定义容器实现执行失败（Executing the custom container implementation failed），提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281455

- **stage-b-test-16-npu-a3**: 日志显示测试运行到88%时，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281485

- **stage-b-test-1-npu-a2 (0)**: 测试套件中3/5通过，但quant目录下的test_npu_autoround_moe.py测试失败，耗时620秒，可能涉及AutoRound量化MoE模型的功能或兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281523

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 31.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281474) |
| stage-b-test-2-npu-a2 (0) | 19.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281558) |
| stage-b-test-2-npu-a2 (1) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281577) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29941187270/job/88995281950) |


## [Run #29940919541](https://github.com/sgl-project/sglang/actions/runs/29940919541)
- **分支**: `mmangkad/flashinfer-0.6.15-post1`
- **总耗时**: 57.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29940919541

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 26.6min | 环境问题 | 自定义容器执行失败，NPU任务在运行中异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402716) |
| multimodal-gen-test-2-npu-a3 | 21.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402733) |
| multimodal-gen-test-1-npu-a3 | 19.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402761) |

- **stage-b-test-4-npu-a3**: 日志显示Prefill批次正常处理，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于NPU环境或容器执行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402716

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions基础设施的警告（如Node 20弃用）和上传artifact时未找到文件的通知，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402733

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402761

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 43.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402751) |
| stage-b-test-2-npu-a2 (0) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402777) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402832) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402919) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994402977) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940919541/job/88994403319) |


## [Run #29940709869](https://github.com/sgl-project/sglang/actions/runs/29940709869)
- **分支**: `mm-per-item-embedding-list`
- **总耗时**: 60.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29940709869

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765385) |
| multimodal-gen-test-2-npu-a3 | 29.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765479) |
| stage-b-test-4-npu-a3 | 26.9min | 环境问题 | 自定义容器执行失败，测试在运行test_npu_tp4_bf16.py时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765681) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示GitHub Actions环境准备、Node.js弃用警告及上传失败产物（无文件）等常规信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765385

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但根本原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765479

- **stage-b-test-4-npu-a3**: 作业在开始第3个测试test_npu_tp4_bf16.py后，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765681

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765437) |
| stage-b-test-1-npu-a2 (1) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765495) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765585) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765651) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993765663) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940709869/job/88993766129) |


## [Run #29940626216](https://github.com/sgl-project/sglang/actions/runs/29940626216)
- **分支**: `kimi-deterministic/6-eagle-seeded-coins`
- **总耗时**: 48.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29940626216

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354377) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354379) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354395) |
| multimodal-gen-test-1-npu-a3 | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354419) |
| multimodal-gen-test-2-npu-a3 | 38.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354442) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354450) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354499) |
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354529) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940626216/job/88993354940) |


## [Run #29940625812](https://github.com/sgl-project/sglang/actions/runs/29940625812)
- **分支**: `kimi-deterministic/3-deepep-normal-bf16-deepgemm`
- **总耗时**: 61.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29940625812

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 31.6min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375030) |
| multimodal-gen-test-2-npu-a3 | 35.2min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375096) |

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常，但中途出现'Executing the custom container implementation failed'错误，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375030

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败断言，仅有checkout、upload-artifact等步骤，且upload-artifact提示无文件上传，可能测试未运行或日志被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375096

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375085) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375101) |
| multimodal-gen-test-1-npu-a3 | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375124) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375135) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375147) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375200) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940625812/job/88993375559) |


## [Run #29940590069](https://github.com/sgl-project/sglang/actions/runs/29940590069)
- **分支**: `ray-metrics`
- **总耗时**: 47.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29940590069

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993248942) |
| multimodal-gen-test-2-npu-a3 | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993248966) |
| multimodal-gen-test-1-npu-a3 | 34.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993249008) |
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993249017) |
| stage-b-test-1-npu-a2 (0) | 44.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993249031) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993249153) |
| stage-b-test-1-npu-a2 (1) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993249342) |
| stage-b-test-2-npu-a2 (0) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993249359) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29940590069/job/88993250034) |


---
*Auto-generated by npu_pr_monitor.py*