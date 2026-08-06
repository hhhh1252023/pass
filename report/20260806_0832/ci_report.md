# NPU CI 执行监控
**生成时间**: 2026-08-06 00:32 UTC
**分析 Run 数**: 34

---

## [Run #31055994575](https://github.com/sgl-project/sglang/actions/runs/31055994575)
- **分支**: `main`
- **总耗时**: 23.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31055994575

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31055994575/job/92473562533) |
| multimodal-gen-test-1-npu-a3 | 20.2min | 环境问题 | 作业因环境问题失败，未找到失败产物文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31055994575/job/92473562551) |

- **multimodal-gen-test-2-npu-a3**: 日志中仅包含Node.js版本弃用警告和diffusion-failures目录无文件上传的提示，未显示测试执行过程或具体失败原因，可能因日志截断或测试未运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31055994575/job/92473562533

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在运行后上传diffusion-failures目录时提示无文件，说明测试未产生失败样本，可能因NPU环境配置或资源问题导致测试未正常执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31055994575/job/92473562551


## [Run #31055241549](https://github.com/sgl-project/sglang/actions/runs/31055241549)
- **分支**: `lsyin/trim-5090-ci-3`
- **总耗时**: 36.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31055241549

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 33.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316888) |
| multimodal-gen-test-2-npu-a3 | 23.4min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316905) |

- **stage-b-test-16-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316888

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，未执行实际测试或出现明确错误，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316905

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316848) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316853) |
| stage-b-test-2-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316868) |
| multimodal-gen-test-1-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316890) |
| stage-b-test-4-npu-a3 (1) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471316895) |
| stage-b-test-1-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471317003) |
| stage-b-test-4-npu-a3 (0) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471317011) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471317176) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055241549/job/92471317260) |


## [Run #31055186449](https://github.com/sgl-project/sglang/actions/runs/31055186449)
- **分支**: `bot/bump-kernel-version-0.4.6-a3f1`
- **总耗时**: 52.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31055186449

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 49.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077159) |
| stage-b-test-4-npu-a3 (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077164) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077175) |
| stage-b-test-2-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077177) |
| stage-b-test-1-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077188) |
| stage-b-test-4-npu-a3 (0) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077248) |
| stage-b-test-8-npu-a3 | 7.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077282) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077496) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31055186449/job/92471077532) |


## [Run #31054783680](https://github.com/sgl-project/sglang/actions/runs/31054783680)
- **分支**: `main`
- **总耗时**: 5.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31054783680

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 4.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957524) |
| stage-b-test-8-npu-a3 | 4.7min | 环境问题 | 自定义容器执行失败，NPU作业在模型加载阶段异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957557) |
| stage-b-test-1-npu-a3 | 4.6min | 环境问题 | NPU测试执行时自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957559) |
| stage-a-unit-test-npu | 4.1min | 环境问题 | 自定义容器执行失败，测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957561) |
| stage-b-test-2-npu-a3 | 3.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957569) |
| stage-b-test-4-npu-a3 (1) | 4.2min | 环境问题 | 自定义容器执行失败，测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957576) |
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957590) |
| stage-b-test-4-npu-a3 (0) | 2.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957603) |
| multimodal-gen-test-2-npu-a3 | 3.5min | 其他 | 作业未执行实际测试，仅上传空产物后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957607) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 2.8min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469958103) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 3.2min | 其他 | 日志被截断，未显示测试执行结果，仅看到作业清理和Node.js弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469958108) |

- **stage-b-test-16-npu-a3**: 作业在启动NPU推理服务时，TokenizerManager和DetokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957524

- **stage-b-test-8-npu-a3**: 作业在初始化模型（ModelSlimW8A8Int8MoE）后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957557

- **stage-b-test-1-npu-a3**: 作业在运行test_npu_autoround_dense.py时，自定义容器执行失败（Executing the custom container implementation failed），可能是NPU环境或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957559

- **stage-a-unit-test-npu**: 作业在启动NPU单元测试时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，导致测试进程未开始即终止，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957561

- **stage-b-test-2-npu-a3**: 作业在运行测试命令后，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957569

- **stage-b-test-4-npu-a3 (1)**: 作业在启动第一个测试时，自定义容器实现执行失败（Executing the custom container implementation failed），导致测试进程未开始即终止，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957576

- **multimodal-gen-test-1-npu-a3**: 日志显示作业正常执行，上传diffusion-failures目录时提示无文件，未发现测试失败或错误信息，可能为作业提前结束或测试未产生失败产物。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957590

- **stage-b-test-4-npu-a3 (0)**: 日志显示在安装sglang_router后，执行自定义容器实现时失败，提示请联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957603

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未发现测试执行记录，且diffusion-failures目录无文件，作业正常完成，无明确失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469957607

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志在测试开始前即结束，缺少实际测试输出和错误信息，无法判断失败原因。可能为日志截断或作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469958103

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志中间部分省略，无法定位具体失败原因。仅显示作业结束时的清理步骤和Node.js 20弃用警告，未包含测试输出或错误信息，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054783680/job/92469958108


## [Run #31054428712](https://github.com/sgl-project/sglang/actions/runs/31054428712)
- **分支**: `main`
- **总耗时**: 6.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31054428712

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 1.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858309) |
| stage-b-test-8-npu-a3 | 4.1min | 环境问题 | 自定义容器执行失败，NPU作业在启动阶段异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858318) |
| stage-b-test-1-npu-a3 | 4.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858354) |
| stage-b-test-16-npu-a3 | 4.2min | 环境问题 | 自定义容器执行失败，NPU分布式初始化未完成 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858360) |
| stage-b-test-4-npu-a3 (1) | 4.8min | 环境问题 | 自定义容器执行失败，可能是容器环境或资源问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858377) |
| stage-b-test-4-npu-a3 (0) | 4.6min | 环境问题 | 自定义容器执行失败，测试未真正运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858393) |
| stage-b-test-2-npu-a3 | 2.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858414) |
| multimodal-gen-test-1-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858460) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 4.5min | 环境问题 | 作业在启动阶段即被终止，未进入实际测试执行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858831) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 1.6min | 其他 | 作业日志不完整，未显示实际测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858863) |

- **multimodal-gen-test-2-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858309

- **stage-b-test-8-npu-a3**: 日志显示作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858318

- **stage-b-test-1-npu-a3**: 作业在运行测试命令时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858354

- **stage-b-test-16-npu-a3**: 作业在TP/EP分布式初始化阶段（TP4 EP4等）启动后，自定义容器实现执行失败，导致作业中止。可能是NPU环境配置或容器问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858360

- **stage-b-test-4-npu-a3 (1)**: 日志显示在安装sglang_router后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858377

- **stage-b-test-4-npu-a3 (0)**: 作业在启动测试后立即报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858393

- **stage-b-test-2-npu-a3**: 日志显示在安装triton-ascend依赖后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858414

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858460

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在准备阶段后立即进入清理流程，未运行测试代码，可能是运行环境或资源分配异常导致作业提前结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858831

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中未包含测试运行的具体输出或错误信息，仅显示runner启动、依赖下载和作业清理步骤，无法判断失败原因，可能为日志截断或作业在启动阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858863

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054428712/job/92468858336) |


## [Run #31054374041](https://github.com/sgl-project/sglang/actions/runs/31054374041)
- **分支**: `yuzhen/fix-blackwell-mla-pack-gqa`
- **总耗时**: 49.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31054374041

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468698885) |
| stage-a-unit-test-npu | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468698886) |
| stage-b-test-16-npu-a3 | 47.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468698888) |
| stage-b-test-8-npu-a3 | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468698890) |
| stage-b-test-2-npu-a3 | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468698902) |
| stage-b-test-1-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468698903) |
| stage-b-test-4-npu-a3 (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468698914) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468699299) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31054374041/job/92468699301) |


## [Run #31053781212](https://github.com/sgl-project/sglang/actions/runs/31053781212)
- **分支**: `lsyin/misc-bugfixes`
- **总耗时**: 17.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31053781212

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 8.9min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893581) |
| stage-b-test-8-npu-a3 | 6.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893603) |
| stage-b-test-4-npu-a3 (1) | 5.0min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893619) |
| multimodal-gen-test-1-npu-a3 | 9.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893622) |
| stage-b-test-1-npu-a3 | 7.1min | 环境问题 | 自定义容器执行失败，runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893623) |
| stage-b-test-4-npu-a3 (0) | 5.0min | 环境问题 | 自定义容器执行失败，NPU图捕获阶段异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893625) |
| multimodal-gen-test-2-npu-a3 | 11.6min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，可能测试未运行或提前退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893633) |
| stage-b-test-16-npu-a3 | 14.3min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893665) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 2.8min | 环境问题 | 测试未生成metrics.json文件，导致性能测试无法完成。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893983) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 8.1min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466894044) |

- **stage-b-test-2-npu-a3**: 日志显示模型权重加载成功（Qwen3MoeForCausalLM），但随后自定义容器实现执行失败，提示联系自托管runner管理员，疑似NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893581

- **stage-b-test-8-npu-a3**: 作业在服务启动后约7秒报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893603

- **stage-b-test-4-npu-a3 (1)**: 作业在加载模型权重时（Multi-thread loading shards 14%）自定义容器实现执行失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893619

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未包含multimodal测试执行的具体输出或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893622

- **stage-b-test-1-npu-a3**: 作业在运行第二个测试文件时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于runner环境或容器配置问题，非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893623

- **stage-b-test-4-npu-a3 (0)**: 作业在NPU图捕获阶段（TP0/TP1/TP2）执行时，自定义容器实现失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893625

- **multimodal-gen-test-2-npu-a3**: 日志显示作业启动后直接进入上传diffusion-failures步骤，但提示无文件可上传，未看到任何测试执行或失败信息，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893633

- **stage-b-test-16-npu-a3**: 作业在加载模型分片至66%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893665

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业在运行性能测试时未找到/tmp/metrics.json文件，无法上传性能指标，测试提前结束。可能是测试脚本未正确执行或环境配置问题导致性能数据未生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893983

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 提供的日志仅包含作业启动、环境准备和清理阶段，未包含测试运行及断言失败信息，因此无法判断是精度、性能还是环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466894044

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053781212/job/92466893602) |


## [Run #31053773174](https://github.com/sgl-project/sglang/actions/runs/31053773174)
- **分支**: `lsyin/trim-5090-ci-3`
- **总耗时**: 12.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31053773174

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 11.1min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831271) |
| stage-b-test-2-npu-a3 | 4.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831276) |
| stage-b-test-4-npu-a3 (0) | 9.1min | 环境问题 | 自定义容器执行失败，模型加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831287) |
| stage-b-test-4-npu-a3 (1) | 8.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831289) |
| multimodal-gen-test-2-npu-a3 | 11.4min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831295) |
| stage-b-test-16-npu-a3 | 10.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831300) |
| multimodal-gen-test-1-npu-a3 | 11.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831319) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.7min | 环境问题 | 作业在启动阶段即被中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831670) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 9.5min | 其他 | 日志不完整，未显示测试执行结果，仅包含作业初始化和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831675) |

- **stage-b-test-1-npu-a3**: 作业在运行第6个测试时，自定义容器实现执行失败，导致测试进程被终止。日志显示测试前5个均通过，失败发生在容器层面而非测试代码本身，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831271

- **stage-b-test-2-npu-a3**: 作业在启动TokenizerManager后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831276

- **stage-b-test-4-npu-a3 (0)**: 日志显示模型分片加载到94%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831287

- **stage-b-test-4-npu-a3 (1)**: 作业在运行第二个测试时，自定义容器实现执行失败，报错提示联系自托管runner管理员，属于基础设施环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831289

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未发现实际测试执行或失败断言，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831295

- **stage-b-test-16-npu-a3**: 日志显示在NPU初始化阶段（DP0/DP1 TP0/TP1）后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831300

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831319

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在准备阶段后立即进入清理流程，无测试执行记录，可能因runner环境异常或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831670

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志在测试运行前被截断，仅包含runner启动、action下载和plog备份等步骤，未出现测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831675

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831293) |
| stage-b-test-8-npu-a3 | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053773174/job/92466831344) |


## [Run #31053350364](https://github.com/sgl-project/sglang/actions/runs/31053350364)
- **分支**: `lsyin/trim-5090-ci-3`
- **总耗时**: 5.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31053350364

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523241) |
| stage-a-unit-test-npu | 4.2min | 环境问题 | NPU测试执行时自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523274) |
| stage-b-test-1-npu-a3 | 4.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523281) |
| stage-b-test-2-npu-a3 | 4.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523293) |
| stage-b-test-8-npu-a3 | 4.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523323) |
| multimodal-gen-test-2-npu-a3 | 4.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523331) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523592) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 4.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523671) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523241

- **stage-a-unit-test-npu**: 作业在运行NPU单元测试时，自定义容器执行失败，错误提示为“Executing the custom container implementation failed”，可能是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523274

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523281

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523293

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523323

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问日志文件时，Azure Blob 返回 BlobNotFound 错误，说明该文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523331

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是资源被清理或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523592

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053350364/job/92465523671


## [Run #31053281784](https://github.com/sgl-project/sglang/actions/runs/31053281784)
- **分支**: `wip/gdn-wy-verify-cudagraph`
- **总耗时**: 44.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31053281784

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (0) | 27.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264940) |
| stage-b-test-16-npu-a3 | 40.9min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264947) |
| multimodal-gen-test-2-npu-a3 | 32.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264968) |

- **stage-b-test-4-npu-a3 (0)**: 日志显示在启动NPU测试容器时，自定义容器实现执行失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264940

- **stage-b-test-16-npu-a3**: 作业在加载模型分片（约69%）时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于运行环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264947

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（未找到文件），未展示测试执行过程或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264968

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264907) |
| stage-b-test-8-npu-a3 | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264913) |
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264935) |
| stage-b-test-2-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264948) |
| stage-b-test-1-npu-a3 | 25.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264953) |
| stage-b-test-4-npu-a3 (1) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465264982) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465265308) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053281784/job/92465265347) |


## [Run #31053215031](https://github.com/sgl-project/sglang/actions/runs/31053215031)
- **分支**: `xpu-mamba-extra-buffer`
- **总耗时**: 51.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31053215031

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 30.0min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032381) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含实际测试执行和失败信息，仅显示Node.js 20弃用警告及上传diffusion-failures工件时未找到文件。测试可能因环境问题或未生成失败文件而提前结束，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032381

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032211) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032212) |
| stage-b-test-2-npu-a3 | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032222) |
| stage-b-test-4-npu-a3 (1) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032242) |
| stage-b-test-16-npu-a3 | 50.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032252) |
| stage-b-test-1-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032291) |
| stage-b-test-4-npu-a3 (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032375) |
| multimodal-gen-test-1-npu-a3 | 35.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032426) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032771) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31053215031/job/92465032848) |


## [Run #31051487751](https://github.com/sgl-project/sglang/actions/runs/31051487751)
- **分支**: `main`
- **总耗时**: 26.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31051487751

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 12.3min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482785) |
| stage-b-test-1-npu-a3 | 21.2min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482786) |
| stage-b-test-2-npu-a3 | 18.8min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482807) |
| stage-b-test-4-npu-a3 (0) | 24.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482820) |
| multimodal-gen-test-2-npu-a3 | 20.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482845) |
| stage-b-test-4-npu-a3 (1) | 24.0min | 环境问题 | 自定义容器执行失败，runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482883) |
| multimodal-gen-test-1-npu-a3 | 22.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482893) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 21.4min | 其他 | 日志被截断，未显示实际测试结果，无法判断失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459483350) |

- **stage-b-test-16-npu-a3**: 日志显示模型分片加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482785

- **stage-b-test-1-npu-a3**: 作业在加载模型权重完成后，执行自定义容器实现时失败，提示联系自托管runner管理员。可能是NPU环境或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482786

- **stage-b-test-2-npu-a3**: 作业在加载模型权重（25%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482807

- **stage-b-test-4-npu-a3 (0)**: 日志显示在运行测试时出现错误："Executing the custom container implementation failed. Please contact your self hosted runner administrator."，表明自托管运行器的容器环境存在问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482820

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行或失败的具体错误，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482845

- **stage-b-test-4-npu-a3 (1)**: 测试运行到第3个用例时，自定义容器实现执行失败，提示联系自托管runner管理员。前2个用例均通过，非代码或性能问题，属于runner环境故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482883

- **multimodal-gen-test-1-npu-a3**: 日志中仅显示Node 20弃用警告、上传diffusion-failures产物时未找到文件，以及清理步骤，未包含测试执行或失败的具体错误信息，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482893

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业启动和清理信息，未包含测试执行过程或错误输出，无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459483350

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482770) |
| stage-b-test-8-npu-a3 | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459482804) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31051487751/job/92459483545) |


## [Run #31050666094](https://github.com/sgl-project/sglang/actions/runs/31050666094)
- **分支**: `main`
- **总耗时**: 11.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31050666094

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 3.1min | 环境问题 | 自定义容器执行失败，测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023433) |
| stage-b-test-2-npu-a3 | 2.3min | 环境问题 | 自定义容器执行失败，构建环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023448) |
| stage-b-test-1-npu-a3 | 10.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023449) |
| stage-b-test-16-npu-a3 | 10.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023476) |
| stage-b-test-4-npu-a3 (0) | 10.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023490) |
| multimodal-gen-test-2-npu-a3 | 1.1min | 环境问题 | 作业在准备阶段因Node.js 20弃用警告而中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023540) |
| stage-b-test-4-npu-a3 (1) | 10.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023634) |
| multimodal-gen-test-1-npu-a3 | 0.6min | 环境问题 | 自托管runner容器启动失败，导致作业无法执行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023659) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.6min | 环境问题 | 作业在启动阶段即失败，未进入实际测试，缺少关键错误日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023905) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 1.1min | 其他 | 作业日志不完整，缺少关键测试执行信息，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023935) |

- **stage-b-test-8-npu-a3**: 作业在启动测试时，自定义容器实现执行失败（Executing the custom container implementation failed），导致测试进程未启动即终止，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023433

- **stage-b-test-2-npu-a3**: 日志显示在创建PEP 517构建环境时，执行setuptools.build_meta.get_requires_for_build_editable()失败，提示自定义容器实现执行失败，需联系自托管runner管理员，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023448

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023449

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023476

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023490

- **multimodal-gen-test-2-npu-a3**: GitHub Actions runner提示Node.js 20已弃用，强制使用Node.js 24运行actions/checkout@v4和actions/upload-artifact@v4，导致作业在初始化阶段失败，未执行多模态生成测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023540

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023634

- **multimodal-gen-test-1-npu-a3**: 日志显示自定义容器实现执行失败，提示jobPod未设置，prepareJob未成功完成。这是runner基础设施问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023659

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在准备阶段后直接进入清理流程，未执行测试步骤，且无metrics.json生成，可能因环境初始化失败或资源分配问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023905

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业初始化和清理阶段，未显示测试运行过程及错误输出，可能因日志截断或作业在启动后立即失败，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023935

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050666094/job/92457023465) |


## [Run #31050454583](https://github.com/sgl-project/sglang/actions/runs/31050454583)
- **分支**: `inkling-small-dgx-spark`
- **总耗时**: 15.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31050454583

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 10.2min | 环境问题 | 自定义容器执行失败，模型加载权重时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344696) |
| stage-b-test-16-npu-a3 | 10.6min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344701) |
| stage-b-test-1-npu-a3 | 10.6min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344705) |
| stage-b-test-4-npu-a3 (0) | 11.3min | 环境问题 | NPU容器在执行模型加载时崩溃，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344707) |
| multimodal-gen-test-1-npu-a3 | 14.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344762) |
| multimodal-gen-test-2-npu-a3 | 14.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344828) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 14.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456345184) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 14.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456345206) |

- **stage-b-test-2-npu-a3**: 作业在加载模型权重（Multi-thread loading shards）时，自定义容器实现执行失败，导致作业终止。日志显示为runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344696

- **stage-b-test-16-npu-a3**: 作业在加载模型分片至85%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344701

- **stage-b-test-1-npu-a3**: 作业在NPU测试执行过程中，自定义容器实现执行失败，日志显示torchair配置警告后作业中断，可能是NPU环境或容器问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344705

- **stage-b-test-4-npu-a3 (0)**: 日志显示模型权重加载到7个分片时进程异常终止，错误为自定义容器执行失败，可能是NPU资源或容器环境不稳定所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344707

- **multimodal-gen-test-1-npu-a3**: 作业日志返回BlobNotFound错误，表明CI流程尝试访问的Azure Blob存储资源缺失或路径错误，可能因文件未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344762

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储账户中缺失，可能是上传失败、路径错误或文件被清理，需检查相关依赖是否正常生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344828

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传、被删除或配置的 URL 有误，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456345184

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456345206

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344672) |
| stage-b-test-8-npu-a3 | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050454583/job/92456344723) |


## [Run #31050375143](https://github.com/sgl-project/sglang/actions/runs/31050375143)
- **分支**: `yuzhen/fix-blackwell-mla-pack-gqa`
- **总耗时**: 62.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31050375143

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 45.8min | 超时 | Scheduler watchdog 超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455968874) |

- **stage-b-test-16-npu-a3**: 日志显示 Scheduler watchdog timeout (self.watchdog_timeout=300)，TP14 EP14 调度器在300秒内无响应，可能因资源竞争或死锁导致，最终作业被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455968874

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455968903) |
| stage-b-test-4-npu-a3 (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455968915) |
| stage-b-test-2-npu-a3 | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455968917) |
| stage-b-test-4-npu-a3 (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455968918) |
| stage-b-test-1-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455968925) |
| stage-b-test-8-npu-a3 | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455968944) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455969689) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31050375143/job/92455969701) |


## [Run #31049386407](https://github.com/sgl-project/sglang/actions/runs/31049386407)
- **分支**: `dsy/spec-eos-beats-length-finish`
- **总耗时**: 67.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31049386407

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452993004) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452993652) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452993004

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452993652

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a3 | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452992906) |
| stage-b-test-1-npu-a3 | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452992909) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452992919) |
| stage-b-test-16-npu-a3 | 53.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452992924) |
| multimodal-gen-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452992966) |
| stage-b-test-8-npu-a3 | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452992973) |
| stage-b-test-4-npu-a3 (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452992988) |
| stage-b-test-4-npu-a3 (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452993026) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049386407/job/92452993654) |


## [Run #31049346424](https://github.com/sgl-project/sglang/actions/runs/31049346424)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 67.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31049346424

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 9.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744539) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 22.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452745025) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744539

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452745025

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a3 | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744471) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744484) |
| stage-b-test-2-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744489) |
| stage-b-test-8-npu-a3 | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744493) |
| stage-b-test-16-npu-a3 | 51.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744507) |
| stage-b-test-4-npu-a3 (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744510) |
| multimodal-gen-test-1-npu-a3 | 25.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744538) |
| stage-b-test-4-npu-a3 (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452744555) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31049346424/job/92452745047) |


## [Run #31048863646](https://github.com/sgl-project/sglang/actions/runs/31048863646)
- **分支**: `main`
- **总耗时**: 25.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31048863646

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 6.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214322) |
| stage-b-test-16-npu-a3 | 6.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214344) |
| stage-b-test-2-npu-a3 | 21.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214363) |
| multimodal-gen-test-2-npu-a3 | 10.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214393) |
| multimodal-gen-test-1-npu-a3 | 9.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214396) |
| stage-b-test-1-npu-a3 | 21.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214415) |
| stage-b-test-4-npu-a3 (1) | 23.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214418) |
| stage-b-test-4-npu-a3 (0) | 23.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214427) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 6.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214754) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214875) |

- **stage-b-test-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214322

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214344

- **stage-b-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214363

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214393

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214396

- **stage-b-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214415

- **stage-b-test-4-npu-a3 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214418

- **stage-b-test-4-npu-a3 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214427

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214754

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214875

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048863646/job/92451214299) |


## [Run #31048504571](https://github.com/sgl-project/sglang/actions/runs/31048504571)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 10.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31048504571

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187575) |
| stage-b-test-2-npu-a3 | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187576) |
| stage-b-test-8-npu-a3 | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187602) |
| stage-b-test-4-npu-a3 (0) | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187606) |
| multimodal-gen-test-2-npu-a3 | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187609) |
| stage-b-test-16-npu-a3 | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187616) |
| stage-b-test-4-npu-a3 (1) | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187625) |
| stage-b-test-1-npu-a3 | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187630) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450188265) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450189139) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187575

- **stage-b-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187576

- **stage-b-test-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187602

- **stage-b-test-4-npu-a3 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187606

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187609

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187616

- **stage-b-test-4-npu-a3 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187625

- **stage-b-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187630

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450188265

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450189139

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048504571/job/92450187557) |


## [Run #31048118485](https://github.com/sgl-project/sglang/actions/runs/31048118485)
- **分支**: `main`
- **总耗时**: 62.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31048118485

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836790) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836790

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 49.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836727) |
| stage-b-test-1-npu-a3 | 25.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836746) |
| stage-b-test-8-npu-a3 | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836748) |
| stage-b-test-4-npu-a3 (1) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836751) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836754) |
| stage-b-test-4-npu-a3 (0) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836910) |
| stage-b-test-2-npu-a3 | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448836916) |
| multimodal-gen-test-1-npu-a3 | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448837105) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448837438) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048118485/job/92448837473) |


## [Run #31048080130](https://github.com/sgl-project/sglang/actions/runs/31048080130)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 6.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31048080130

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766699) |
| stage-a-unit-test-npu | 4.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766718) |
| stage-b-test-16-npu-a3 | 3.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766762) |
| stage-b-test-2-npu-a3 | 4.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766764) |
| stage-b-test-4-npu-a3 (0) | 4.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766779) |
| stage-b-test-4-npu-a3 (1) | 4.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766796) |
| stage-b-test-8-npu-a3 | 2.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766818) |
| stage-b-test-1-npu-a3 | 4.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766913) |
| multimodal-gen-test-2-npu-a3 | 4.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766942) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448767273) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 4.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448767285) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766699

- **stage-a-unit-test-npu**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766718

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766762

- **stage-b-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766764

- **stage-b-test-4-npu-a3 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766779

- **stage-b-test-4-npu-a3 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766796

- **stage-b-test-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766818

- **stage-b-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766913

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448766942

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448767273

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048080130/job/92448767285


## [Run #31048015639](https://github.com/sgl-project/sglang/actions/runs/31048015639)
- **分支**: `main`
- **总耗时**: 10.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31048015639

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 4.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681722) |
| stage-b-test-1-npu-a3 | 4.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681726) |
| stage-b-test-4-npu-a3 (0) | 6.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681732) |
| stage-b-test-4-npu-a3 (1) | 6.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681745) |
| stage-b-test-8-npu-a3 | 6.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681753) |
| stage-b-test-16-npu-a3 | 6.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681761) |
| multimodal-gen-test-1-npu-a3 | 6.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681770) |
| multimodal-gen-test-2-npu-a3 | 6.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681817) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448682463) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 7.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448682516) |

- **stage-b-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681722

- **stage-b-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681726

- **stage-b-test-4-npu-a3 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681732

- **stage-b-test-4-npu-a3 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681745

- **stage-b-test-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681753

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681761

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681770

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681817

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448682463

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448682516

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448493989) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31048015639/job/92448681718) |


## [Run #31047676328](https://github.com/sgl-project/sglang/actions/runs/31047676328)
- **分支**: `main`
- **总耗时**: 5.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31047676328

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 4.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281581) |
| stage-b-test-4-npu-a3 (0) | 1.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281621) |
| stage-b-test-16-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281639) |
| multimodal-gen-test-1-npu-a3 | 4.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281641) |
| stage-b-test-8-npu-a3 | 1.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281648) |
| stage-a-unit-test-npu | 3.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281651) |
| stage-b-test-4-npu-a3 (1) | 1.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281662) |
| stage-b-test-1-npu-a3 | 4.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281702) |
| multimodal-gen-test-2-npu-a3 | 4.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281738) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447282190) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 4.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447282194) |

- **stage-b-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281581

- **stage-b-test-4-npu-a3 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281621

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281639

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281641

- **stage-b-test-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281648

- **stage-a-unit-test-npu**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281651

- **stage-b-test-4-npu-a3 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281662

- **stage-b-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281702

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447281738

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447282190

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31047676328/job/92447282194


## [Run #31046297061](https://github.com/sgl-project/sglang/actions/runs/31046297061)
- **分支**: `main`
- **总耗时**: 19.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31046297061

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693332) |
| stage-b-test-4-npu-a3 (0) | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693349) |
| stage-b-test-2-npu-a3 | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693356) |
| stage-b-test-4-npu-a3 (1) | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693357) |
| multimodal-gen-test-2-npu-a3 | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693366) |
| multimodal-gen-test-1-npu-a3 | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693390) |
| stage-b-test-1-npu-a3 | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693501) |
| stage-b-test-8-npu-a3 | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693598) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442694015) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 17.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442694103) |

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693332

- **stage-b-test-4-npu-a3 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693349

- **stage-b-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693356

- **stage-b-test-4-npu-a3 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693357

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693366

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693390

- **stage-b-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693501

- **stage-b-test-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693598

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442694015

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442694103

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046297061/job/92442693495) |


## [Run #31046162561](https://github.com/sgl-project/sglang/actions/runs/31046162561)
- **分支**: `bot/bump-kernel-version-0.4.6-a3f1`
- **总耗时**: 75.6min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31046162561

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 51.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442173020) |
| stage-b-test-4-npu-a3 (1) | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442173094) |
| stage-b-test-8-npu-a3 | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442173116) |
| stage-b-test-4-npu-a3 (0) | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442173119) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442173122) |
| stage-b-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442173127) |
| stage-b-test-2-npu-a3 | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442173143) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442174029) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31046162561/job/92442174062) |


## [Run #31044893268](https://github.com/sgl-project/sglang/actions/runs/31044893268)
- **分支**: `qiaolin_spec_mrope_opt`
- **总耗时**: 72.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31044893268

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993461) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993461

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 (1) | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993384) |
| stage-b-test-8-npu-a3 | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993403) |
| stage-b-test-16-npu-a3 | 55.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993417) |
| stage-b-test-2-npu-a3 | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993431) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993462) |
| multimodal-gen-test-1-npu-a3 | 25.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993466) |
| stage-b-test-1-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993467) |
| stage-b-test-4-npu-a3 (0) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437993486) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437994180) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31044893268/job/92437994266) |


## [Run #31043830559](https://github.com/sgl-project/sglang/actions/runs/31043830559)
- **分支**: `main`
- **总耗时**: 29.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31043830559

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 28.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31043830559/job/92434583946) |
| multimodal-gen-test-2-npu-a3 | 28.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31043830559/job/92434584018) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31043830559/job/92434583946

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31043830559/job/92434584018


## [Run #31043691823](https://github.com/sgl-project/sglang/actions/runs/31043691823)
- **分支**: `refactor-mxfp4-sm100-trtllm-moerunner`
- **总耗时**: 85.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31043691823

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058737) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434059226) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058737

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434059226

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058542) |
| multimodal-gen-test-1-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058563) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058587) |
| stage-b-test-4-npu-a3 (1) | 16.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058600) |
| stage-b-test-4-npu-a3 (0) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058622) |
| stage-b-test-2-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058633) |
| stage-b-test-1-npu-a3 | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058668) |
| stage-b-test-16-npu-a3 | 57.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434058727) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043691823/job/92434059311) |


## [Run #31043440090](https://github.com/sgl-project/sglang/actions/runs/31043440090)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 57.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31043440090

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 35.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187551) |
| multimodal-gen-test-2-npu-a3 | 8.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187628) |
| multimodal-gen-test-1-npu-a3 | 17.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187655) |

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187551

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187628

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187655

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187549) |
| stage-b-test-8-npu-a3 | 7.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187565) |
| stage-b-test-2-npu-a3 | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187631) |
| stage-b-test-1-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187647) |
| stage-b-test-4-npu-a3 (1) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187778) |
| stage-b-test-4-npu-a3 (0) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433187792) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433188035) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31043440090/job/92433188081) |


## [Run #31042663336](https://github.com/sgl-project/sglang/actions/runs/31042663336)
- **分支**: `qiaolin_fused_commit_indices`
- **总耗时**: 84.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31042663336

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 22.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675286) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675663) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675286

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675663

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675176) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675181) |
| multimodal-gen-test-1-npu-a3 | 28.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675190) |
| stage-b-test-8-npu-a3 | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675239) |
| stage-b-test-2-npu-a3 | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675252) |
| stage-b-test-4-npu-a3 (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675253) |
| stage-b-test-4-npu-a3 (1) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675255) |
| stage-b-test-16-npu-a3 | 55.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675270) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31042663336/job/92430675757) |


## [Run #31041253679](https://github.com/sgl-project/sglang/actions/runs/31041253679)
- **分支**: `agent/fix-dcp-kv-head-mapping`
- **总耗时**: 88.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31041253679

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969643) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969643

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 52.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969567) |
| stage-b-test-1-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969588) |
| multimodal-gen-test-1-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969591) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969610) |
| stage-b-test-2-npu-a3 | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969637) |
| stage-b-test-8-npu-a3 | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969668) |
| stage-b-test-4-npu-a3 (0) | 33.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969693) |
| stage-b-test-4-npu-a3 (1) | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425969741) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425970413) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31041253679/job/92425970457) |


## [Run #31040836891](https://github.com/sgl-project/sglang/actions/runs/31040836891)
- **分支**: `patch-3`
- **总耗时**: 91.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31040836891

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967211) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967211

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 50.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967190) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967197) |
| stage-b-test-2-npu-a3 | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967212) |
| stage-b-test-1-npu-a3 | 25.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967223) |
| stage-b-test-8-npu-a3 | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967230) |
| multimodal-gen-test-1-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967242) |
| stage-b-test-4-npu-a3 (1) | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967254) |
| stage-b-test-4-npu-a3 (0) | 32.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967320) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967843) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040836891/job/92424967900) |


## [Run #31040622666](https://github.com/sgl-project/sglang/actions/runs/31040622666)
- **分支**: `main`
- **总耗时**: 41.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31040622666

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 30.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942024) |
| stage-b-test-4-npu-a3 (0) | 14.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942040) |
| stage-b-test-4-npu-a3 (1) | 13.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942049) |
| stage-b-test-1-npu-a3 | 17.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942060) |
| stage-b-test-8-npu-a3 | 13.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942091) |
| stage-b-test-16-npu-a3 | 22.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942113) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 18.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942717) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 35.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942786) |

- **stage-b-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942024

- **stage-b-test-4-npu-a3 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942040

- **stage-b-test-4-npu-a3 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942049

- **stage-b-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942060

- **stage-b-test-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942091

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942113

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942717

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942786

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040622666/job/92423942025) |


## [Run #31040593622](https://github.com/sgl-project/sglang/actions/runs/31040593622)
- **分支**: `patch-2`
- **总耗时**: 93.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31040593622

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101279) |
| stage-b-test-16-npu-a3 | 1.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101434) |

- **multimodal-gen-test-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101279

- **stage-b-test-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101434

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101257) |
| multimodal-gen-test-1-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101305) |
| stage-b-test-1-npu-a3 | 25.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101322) |
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101340) |
| stage-b-test-4-npu-a3 (0) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101343) |
| stage-b-test-8-npu-a3 | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101368) |
| stage-b-test-2-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101380) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101976) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31040593622/job/92424101980) |


---
*Auto-generated by npu_pr_monitor.py*