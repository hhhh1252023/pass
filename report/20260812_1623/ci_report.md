# NPU CI 执行监控
**生成时间**: 2026-08-12 08:23 UTC
**分析 Run 数**: 27

---

## [Run #30601504810](https://github.com/sgl-project/sglang/actions/runs/30601504810)
- **分支**: `lsyin/verify-buffer-api`
- **总耗时**: 13.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30601504810

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 11.3min | 环境问题 | 自定义容器执行失败，NPU测试中途崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099443) |
| stage-b-test-4-npu-a3 | 11.0min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099471) |
| multimodal-gen-test-2-npu-a3 | 10.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099488) |
| stage-b-test-2-npu-a2 (0) | 11.7min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099491) |
| multimodal-gen-test-1-npu-a3 | 11.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099501) |
| stage-b-test-1-npu-a2 (1) | 11.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099507) |
| stage-b-test-2-npu-a2 (1) | 11.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099509) |
| stage-b-test-1-npu-a2 (0) | 11.8min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099529) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.6min | 其他 | 日志被截断，未显示测试执行失败的具体原因，仅见清理和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099793) |

- **stage-b-test-16-npu-a3**: 日志显示在DeepSeek-R1模型测试进行到Decode阶段时，自定义容器实现执行失败，导致作业终止。可能是NPU资源问题或容器环境不稳定，并非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099443

- **stage-b-test-4-npu-a3**: 日志显示模型加载过程中出现tokenizer组件修复和NumPy警告，随后自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099471

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有GitHub Actions环境准备、Node.js弃用警告及上传artifact时无文件等提示，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099488

- **stage-b-test-2-npu-a2 (0)**: 测试运行到第二个用例时，GitHub Actions 报错“Executing the custom container implementation failed”，属于自托管 runner 容器环境异常，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099491

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体错误或失败信息，仅显示Node 20弃用警告和上传artifact时无文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099501

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常，但中途出现“Executing the custom container implementation failed”错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099507

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行中（进度42%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099509

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常（Accuracy 0.868），但在开始第二个测试时，runner报错"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099529

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分省略，无法定位失败点。作业在运行后进入plog备份和清理阶段，未输出metrics.json，可能因测试未产生结果或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099793

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30601504810/job/91065099465) |


## [Run #30601324104](https://github.com/sgl-project/sglang/actions/runs/30601324104)
- **分支**: `main`
- **总耗时**: 6.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30601324104

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 4.9min | 环境问题 | 测试通过但自定义容器执行失败，属于自托管runner环境问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485055) |
| stage-b-test-1-npu-a2 (1) | 4.7min | 环境问题 | 自定义容器执行失败，导致测试未运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485058) |
| stage-b-test-1-npu-a2 (0) | 4.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485062) |
| stage-b-test-2-npu-a2 (0) | 5.5min | 环境问题 | 自定义容器执行失败，测试在启动后立即中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485064) |
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业未显示明确失败原因，日志仅包含正常执行和Node 20弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485065) |
| stage-b-test-16-npu-a3 | 2.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485078) |
| multimodal-gen-test-2-npu-a3 | 5.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485079) |
| stage-b-test-2-npu-a2 (1) | 5.0min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485087) |
| stage-b-test-4-npu-a3 | 1.7min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485088) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.0min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485176) |

- **stage-a-unit-test-npu**: NPU单元测试50项全部通过，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器执行环境故障，非代码问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485055

- **stage-b-test-1-npu-a2 (1)**: 日志显示在运行测试命令后，自定义容器实现执行失败（Executing the custom container implementation failed），提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485058

- **stage-b-test-1-npu-a2 (0)**: 作业在运行HiCache MHA测试时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU环境或容器配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485062

- **stage-b-test-2-npu-a2 (0)**: 作业在运行test_npu_graph_tp2_bf16.py时，自定义容器实现执行失败（Executing the custom container implementation failed），导致测试进程异常终止，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485064

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败、超时或错误信息，仅显示上传artifact时无文件（diffusion-failures/为空），可能测试全部通过或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485065

- **stage-b-test-16-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于基础设施或容器配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485078

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但根本原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485079

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试刚开始运行（test_npu_mla_fia_w8a8int8.py），随后出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485087

- **stage-b-test-4-npu-a3**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业中断，属于环境或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485088

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业初始化和清理步骤，未展示测试运行输出或错误信息，无法判断失败原因。可能为日志收集不完整或测试未实际执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601324104/job/91064485176


## [Run #30601029590](https://github.com/sgl-project/sglang/actions/runs/30601029590)
- **分支**: `lsyin/verify-buffer-api`
- **总耗时**: 11.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30601029590

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 10.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558639) |
| stage-b-test-1-npu-a2 (0) | 9.9min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558640) |
| multimodal-gen-test-1-npu-a3 | 10.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558656) |
| multimodal-gen-test-2-npu-a3 | 10.9min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558664) |
| stage-b-test-4-npu-a3 | 9.3min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558665) |
| stage-b-test-2-npu-a2 (0) | 9.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558668) |
| stage-b-test-1-npu-a2 (1) | 10.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558669) |
| stage-b-test-2-npu-a2 (1) | 9.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558707) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.7min | 其他 | 作业日志被截断，未显示实际失败原因，仅见清理和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558946) |

- **stage-b-test-16-npu-a3**: 作业在启动NPU多进程（TP/EP）时，各进程获取ASCEND_OPP_PATH后，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境配置或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558639

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到66%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU资源或容器环境问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558640

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558656

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。作业在运行约10分钟后结束，上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558664

- **stage-b-test-4-npu-a3**: 日志显示测试运行到56%时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是容器或runner环境问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558665

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行中突然报错“Executing the custom container implementation failed”，随后作业终止，无测试断言失败或性能异常，属于runner容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558668

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558669

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常（HTTP 200，进度19%），但突然报错“Executing the custom container implementation failed”，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558707

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分省略，末尾仅显示plog备份、Node警告和清理流程，未包含测试执行或失败错误信息，无法判断具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558946

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30601029590/job/91063558713) |


## [Run #30599456594](https://github.com/sgl-project/sglang/actions/runs/30599456594)
- **分支**: `ulysses-ipc-a2a-2rank`
- **总耗时**: 29.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30599456594

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.9min | 其他 | 作业正常结束，无失败迹象，仅上传工件时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30599456594/job/91058912752) |

- **multimodal-gen-test-2-npu-a3**: 日志显示作业成功完成，仅在上传diffusion-failures工件时提示无文件，属正常情况。作业未报告任何测试失败或错误，可能因测试全部通过而无需上传失败日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30599456594/job/91058912752

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599456594/job/91058912709) |


## [Run #30599366694](https://github.com/sgl-project/sglang/actions/runs/30599366694)
- **分支**: `fix-dsa-sparse-prefill-topk-length`
- **总耗时**: 46.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30599366694

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 44.6min | 其他 | 作业日志被截断，未显示实际失败原因，仅见上传了diffusion-failures工件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604815) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分省略，无法定位具体错误。从结尾看，作业上传了diffusion-failures-npu-2-1.zip，表明测试存在失败案例，但具体失败原因需查看该工件或完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604815

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604789) |
| stage-b-test-1-npu-a2 (0) | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604792) |
| stage-b-test-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604795) |
| stage-b-test-2-npu-a2 (1) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604798) |
| stage-b-test-1-npu-a2 (1) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604799) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604813) |
| multimodal-gen-test-1-npu-a3 | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604865) |
| stage-b-test-16-npu-a3 | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058604870) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30599366694/job/91058605083) |


## [Run #30598638454](https://github.com/sgl-project/sglang/actions/runs/30598638454)
- **分支**: `fangyuan/dspark`
- **总耗时**: 27.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30598638454

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 25.5min | 环境问题 | 自定义容器执行失败，测试进程异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425359) |
| stage-b-test-1-npu-a2 (0) | 26.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425370) |
| multimodal-gen-test-2-npu-a3 | 27.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425371) |
| multimodal-gen-test-1-npu-a3 | 25.5min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425379) |
| stage-b-test-1-npu-a2 (1) | 26.6min | 环境问题 | 自定义容器执行失败，NPU显存不足导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425401) |

- **stage-b-test-4-npu-a3**: 在运行test_npu_mla_w8a8int8.py测试时，自定义容器实现执行失败，导致作业中断。可能是NPU环境或容器配置问题，而非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425359

- **stage-b-test-1-npu-a2 (0)**: 日志显示模型权重加载到75%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425370

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、checkout、upload-artifact等步骤，未出现测试执行或失败断言。可能因日志截断或作业在测试前被取消，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425371

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425379

- **stage-b-test-1-npu-a2 (1)**: 日志显示在捕获批次时可用显存仅约10.5GB，且逐步降低batch size仍无法完成，最终容器实现执行失败，属于NPU资源不足的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425401

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425369) |
| stage-b-test-2-npu-a2 (1) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425376) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425385) |
| stage-a-unit-test-npu | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425397) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30598638454/job/91056425623) |


## [Run #30597868458](https://github.com/sgl-project/sglang/actions/runs/30597868458)
- **分支**: `lsyin/verify-buffer-api`
- **总耗时**: 43.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30597868458

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 34.5min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138765) |
| multimodal-gen-test-2-npu-a3 | 33.2min | 精度回归 | 多模态生成测试失败，上传了diffusion-failures工件 | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138811) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业执行了上传diffusion-failures目录的步骤，但提示未找到该目录，无其他错误信息。可能测试未产生失败文件或测试本身未运行，需查看更完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138765

- **multimodal-gen-test-2-npu-a3**: 作业在NPU A3上运行多模态生成测试，最终上传了diffusion-failures-npu-2-1.zip工件，表明存在diffusion模型生成结果与预期不符的精度问题，属于精度回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138811

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138749) |
| stage-b-test-16-npu-a3 | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138758) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138768) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138782) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138792) |
| stage-b-test-4-npu-a3 | 35.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138798) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054138799) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597868458/job/91054139101) |


## [Run #30597699776](https://github.com/sgl-project/sglang/actions/runs/30597699776)
- **分支**: `dev/fanshuaishuai/feat_overlap_image_load`
- **总耗时**: 45.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30597699776

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 40.4min | 精度回归 | 多模态生成测试失败，上传了diffusion-failures-npu-2-1工件，表明存在精度或输出不匹配问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628527) |

- **multimodal-gen-test-2-npu-a3**: 作业在NPU A3上运行多模态生成测试，最终上传了名为diffusion-failures-npu-2-1的失败工件，说明测试中diffusion模型输出与预期不符，属于精度回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628527

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628451) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628454) |
| stage-b-test-1-npu-a2 (1) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628456) |
| stage-b-test-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628471) |
| multimodal-gen-test-1-npu-a3 | 27.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628472) |
| stage-b-test-16-npu-a3 | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628473) |
| stage-b-test-1-npu-a2 (0) | 43.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628483) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628488) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30597699776/job/91053628788) |


## [Run #30596753914](https://github.com/sgl-project/sglang/actions/runs/30596753914)
- **分支**: `feature/eplb-a2a-none-rank-invariant-dispatch`
- **总耗时**: 58.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30596753914

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 34.4min | 其他 | 作业上传了diffusion-failures工件，表明测试存在失败用例，但日志未显示具体错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821315) |
| stage-b-test-4-npu-a3 | 11.1min | 超时 | HiCache MLA测试运行超时且失败，测试耗时440秒超过预估400秒。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821349) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-1.zip工件，说明multimodal生成测试中有失败案例，但未提供具体失败原因，需下载工件进一步分析。
  链接: https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821315

- **stage-b-test-4-npu-a3**: test_npu_hicache_mla.py测试在424秒后失败，超出预估时间，导致整体测试0/5通过。可能是性能回归或环境问题导致测试超时。
  链接: https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821349

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821313) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821318) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821319) |
| multimodal-gen-test-1-npu-a3 | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821327) |
| stage-b-test-16-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821344) |
| stage-b-test-2-npu-a2 (1) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821347) |
| stage-b-test-1-npu-a2 (0) | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821382) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596753914/job/91050821739) |


## [Run #30596406249](https://github.com/sgl-project/sglang/actions/runs/30596406249)
- **分支**: `fix/hicache-mamba-chunk-boundary-backup`
- **总耗时**: 54.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30596406249

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 33.5min | 代码错误 | NPU测试中test_npu_autoround_moe.py失败，退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674652) |
| multimodal-gen-test-2-npu-a3 | 42.1min | 其他 | 作业日志被截断，仅显示上传失败产物，未提供具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674669) |

- **stage-b-test-1-npu-a2 (0)**: 该测试用例在Ascend NPU上执行失败，3/5通过，2个失败，其中quant相关测试失败，可能涉及量化功能或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674652

- **multimodal-gen-test-2-npu-a3**: 日志中仅包含GitHub Actions基础设施信息（Node版本警告、artifact上传成功），未显示测试执行或失败的具体错误。作业可能因测试失败但日志未捕获，或失败发生在被省略的中间部分。
  链接: https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674669

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674629) |
| stage-b-test-4-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674633) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674635) |
| stage-b-test-1-npu-a2 (1) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674654) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674655) |
| stage-b-test-16-npu-a3 | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674673) |
| stage-b-test-2-npu-a2 (0) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674680) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30596406249/job/91049674858) |


## [Run #30595771186](https://github.com/sgl-project/sglang/actions/runs/30595771186)
- **分支**: `ulysses-ipc-a2a-2rank`
- **总耗时**: 40.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30595771186

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 27.0min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595771186/job/91047804814) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。结尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595771186/job/91047804814

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 36.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595771186/job/91047804797) |


## [Run #30595675197](https://github.com/sgl-project/sglang/actions/runs/30595675197)
- **分支**: `agent/fastsafetensors-disable-gds`
- **总耗时**: 47.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30595675197

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 23.5min | 环境问题 | 作业因缺少失败产物文件而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480692) |
| stage-b-test-1-npu-a2 (0) | 44.5min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480705) |
| stage-b-test-4-npu-a3 | 10.7min | 超时 | NPU测试用例test_npu_mla_w8a8int8.py执行超时且失败，导致整个作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480732) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures目录时提示无文件，说明测试未产生失败样本，作业可能因前置条件未满足或测试未运行而终止，属于环境或流程配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480692

- **stage-b-test-1-npu-a2 (0)**: 测试运行到第4个用例时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480705

- **stage-b-test-4-npu-a3**: 测试文件test_npu_mla_w8a8int8.py运行428秒后返回退出码1，超过预估的400秒，且测试摘要显示0/5通过，属于测试超时或执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480732

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480682) |
| stage-b-test-2-npu-a2 (0) | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480697) |
| stage-b-test-16-npu-a3 | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480707) |
| multimodal-gen-test-1-npu-a3 | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480709) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480711) |
| stage-b-test-1-npu-a2 (1) | 33.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480720) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595675197/job/91047480882) |


## [Run #30595540394](https://github.com/sgl-project/sglang/actions/runs/30595540394)
- **分支**: `fix-teardown-gpu-idle`
- **总耗时**: 51.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30595540394

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 46.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595540394/job/91047067164) |
| stage-b-test-1-npu-a2 (1) | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595540394/job/91047067171) |
| stage-b-test-2-npu-a2 (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595540394/job/91047067173) |
| stage-a-unit-test-npu | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595540394/job/91047067176) |
| stage-b-test-16-npu-a3 | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595540394/job/91047067186) |
| stage-b-test-2-npu-a2 (1) | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595540394/job/91047067194) |
| stage-b-test-4-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595540394/job/91047067217) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595540394/job/91047067432) |


## [Run #30595455234](https://github.com/sgl-project/sglang/actions/runs/30595455234)
- **分支**: `ulysses-ipc-a2a-2rank`
- **总耗时**: 7.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30595455234

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595455234/job/91046829991) |
| multimodal-gen-test-2-npu-a3 | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595455234/job/91046830006) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595455234/job/91046829991

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595455234/job/91046830006


## [Run #30595390738](https://github.com/sgl-project/sglang/actions/runs/30595390738)
- **分支**: `feature/eplb-a2a-none-rank-invariant-dispatch`
- **总耗时**: 31.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30595390738

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 0.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658070) |
| stage-b-test-4-npu-a3 | 20.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658072) |
| stage-a-unit-test-npu | 10.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658084) |
| multimodal-gen-test-1-npu-a3 | 22.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658104) |
| stage-b-test-1-npu-a2 (1) | 25.3min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658111) |
| stage-b-test-1-npu-a2 (0) | 29.3min | 环境问题 | CUDA graph捕获阶段容器执行失败，疑似NPU环境不稳定或资源限制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658116) |
| multimodal-gen-test-2-npu-a3 | 15.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658153) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.5min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658417) |

- **stage-b-test-16-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，报错提示联系自托管runner管理员，属于基础设施或容器配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658070

- **stage-b-test-4-npu-a3**: 日志显示测试正常运行中，但突然报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658072

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败、过期或被误删，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658084

- **multimodal-gen-test-1-npu-a3**: 日志截断，缺少测试执行和失败断言部分。仅看到上传artifact时无文件，说明测试可能未运行或提前退出，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658104

- **stage-b-test-1-npu-a2 (1)**: 作业在加载模型权重时，自定义容器实现执行失败，导致进程终止。日志显示权重加载刚开始即报错，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658111

- **stage-b-test-1-npu-a2 (0)**: 日志显示模型加载和KV Cache分配正常，但在CUDA graph捕获（num_tokens=3968）时容器异常退出，错误为自定义容器执行失败，可能因NPU驱动或显存不足导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658116

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions初始化、checkout和upload-artifact步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658153

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业初始化和清理阶段，未展示测试运行输出或错误信息，可能因日志截断或测试未实际执行导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658417

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658080) |
| stage-b-test-2-npu-a2 (0) | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595390738/job/91046658085) |


## [Run #30595159762](https://github.com/sgl-project/sglang/actions/runs/30595159762)
- **分支**: `amd_helios`
- **总耗时**: 42.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30595159762

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 33.5min | 精度回归 | 多模态生成测试失败，上传了diffusion-failures-npu-2-1工件，表明存在精度或输出不一致问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927904) |

- **multimodal-gen-test-2-npu-a3**: 作业在NPU A3上运行多模态生成测试，最终上传了diffusion-failures工件，说明测试过程中产生了失败样本，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927904

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927826) |
| stage-b-test-1-npu-a2 (1) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927828) |
| stage-b-test-16-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927839) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927843) |
| multimodal-gen-test-1-npu-a3 | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927854) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927866) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927880) |
| stage-b-test-2-npu-a2 (1) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045927990) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595159762/job/91045928317) |


## [Run #30595124611](https://github.com/sgl-project/sglang/actions/runs/30595124611)
- **分支**: `feature/eplb-a2a-none-rank-invariant-dispatch`
- **总耗时**: 5.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30595124611

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 4.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967290) |
| stage-a-unit-test-npu | 2.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967305) |
| multimodal-gen-test-1-npu-a3 | 4.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967321) |
| multimodal-gen-test-2-npu-a3 | 4.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967354) |
| stage-b-test-4-npu-a3 | 4.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967357) |
| stage-b-test-1-npu-a2 (1) | 2.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967363) |
| stage-b-test-1-npu-a2 (0) | 1.2min | 环境问题 | 自定义容器执行失败，导致作业在环境准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967376) |
| stage-b-test-2-npu-a2 (1) | 1.4min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967380) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.0min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967666) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967290

- **stage-a-unit-test-npu**: 日志显示在安装依赖后，执行自定义容器实现时失败，提示联系自托管runner管理员。可能是容器环境配置或资源问题，而非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967305

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967321

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件或数据在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967354

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或存储配置变更，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967357

- **stage-b-test-1-npu-a2 (1)**: 日志显示在安装依赖（如numpy、psutil等）后，执行自定义容器实现时失败，错误提示为“Executing the custom container implementation failed”，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967363

- **stage-b-test-1-npu-a2 (0)**: 日志显示在安装系统包（如ca-certificates）后，出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967376

- **stage-b-test-2-npu-a2 (1)**: 日志显示在安装uv后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967380

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业尝试从Azure Blob下载日志文件，但返回BlobNotFound错误，说明该文件已被删除或路径错误，属于CI基础设施或存储配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595124611/job/91045967666


## [Run #30595007873](https://github.com/sgl-project/sglang/actions/runs/30595007873)
- **分支**: `fix/online-mxfp8-linear`
- **总耗时**: 44.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30595007873

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 33.3min | 其他 | 作业上传了diffusion-failures工件，表明测试存在失败用例，但日志未显示具体错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474129) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-1.zip工件，说明multimodal生成测试中有失败案例，但未在日志中展示具体错误信息，需下载工件查看详细失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474129

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474158) |
| stage-b-test-16-npu-a3 | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474167) |
| stage-b-test-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474182) |
| stage-b-test-1-npu-a2 (1) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474192) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474196) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474211) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474215) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474216) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30595007873/job/91045474454) |


## [Run #30594825559](https://github.com/sgl-project/sglang/actions/runs/30594825559)
- **分支**: `feature/eplb-a2a-none-rank-invariant-dispatch`
- **总耗时**: 7.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30594825559

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 2.4min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865810) |
| stage-b-test-2-npu-a2 (1) | 5.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化或资源分配异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865818) |
| multimodal-gen-test-2-npu-a3 | 1.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865824) |
| stage-b-test-2-npu-a2 (0) | 5.7min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865830) |
| stage-b-test-1-npu-a2 (1) | 4.6min | 环境问题 | 自定义容器执行失败，测试未开始即中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865841) |
| stage-b-test-1-npu-a2 (0) | 4.8min | 环境问题 | 自定义容器执行失败，测试启动后runner异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865842) |
| stage-b-test-4-npu-a3 | 6.1min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865847) |
| stage-a-unit-test-npu | 4.4min | 环境问题 | NPU测试执行时自定义容器运行失败，导致测试未完成。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865849) |
| stage-b-test-16-npu-a3 | 2.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865876) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044866302) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行测试后上传diffusion-failures目录时提示无文件，但关键测试输出被省略，无法判断具体失败点。需查看完整日志确认是测试断言失败还是环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865810

- **stage-b-test-2-npu-a2 (1)**: 日志显示模型权重加载成功，但在获取ASCEND_OPP_PATH后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865818

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865824

- **stage-b-test-2-npu-a2 (0)**: 日志显示模型权重加载到50%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865830

- **stage-b-test-1-npu-a2 (1)**: 作业在启动第一个测试后立即报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865841

- **stage-b-test-1-npu-a2 (0)**: 测试test_npu_hicache_mha.py刚开始执行（test_a_gsm8k），约30秒后runner报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865842

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865847

- **stage-a-unit-test-npu**: 日志显示测试开始后约15秒，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于NPU环境或容器配置问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865849

- **stage-b-test-16-npu-a3**: 作业在安装系统依赖包后，执行自定义容器实现时失败，报错提示联系自托管runner管理员，属于runner环境配置或容器启动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044865876

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明CI系统尝试下载或访问的blob资源缺失，可能是构建产物或依赖文件未正确上传，或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594825559/job/91044866302


## [Run #30594683801](https://github.com/sgl-project/sglang/actions/runs/30594683801)
- **分支**: `main`
- **总耗时**: 22.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30594683801

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486559) |
| stage-b-test-16-npu-a3 | 11.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486566) |
| stage-b-test-1-npu-a2 (0) | 20.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486599) |
| stage-b-test-4-npu-a3 | 12.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486612) |
| stage-b-test-2-npu-a2 (1) | 21.4min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486616) |
| stage-b-test-1-npu-a2 (1) | 20.3min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486627) |
| multimodal-gen-test-2-npu-a3 | 16.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含上传失败产物和清理步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486647) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.6min | 环境问题 | 测试未生成metrics.json文件，导致性能测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044487119) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486559

- **stage-b-test-16-npu-a3**: 日志显示测试运行正常（HTTP 200），但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486566

- **stage-b-test-1-npu-a2 (0)**: 作业在启动NPU推理服务时，自定义容器实现执行失败，日志显示TokenizerManager初始化后即报错，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486599

- **stage-b-test-4-npu-a3**: 日志显示在捕获批次过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486612

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常（进度74%），但突然报错“Executing the custom container implementation failed”，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486616

- **stage-b-test-1-npu-a2 (1)**: 日志显示在加载模型权重时（Multi-thread loading shards 0%），自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486627

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败断言信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或提前退出，需查看完整日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486647

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业在运行性能测试后未找到/tmp/metrics.json文件，无法上传性能指标。日志显示测试流程正常执行但未产出预期结果，可能是测试脚本执行失败或环境配置问题导致性能数据未生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044487119

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486557) |
| stage-b-test-2-npu-a2 (0) | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30594683801/job/91044486598) |


## [Run #30594432066](https://github.com/sgl-project/sglang/actions/runs/30594432066)
- **分支**: `main`
- **总耗时**: 5.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30594432066

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 4.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718211) |
| multimodal-gen-test-2-npu-a3 | 4.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718216) |
| stage-b-test-1-npu-a2 (0) | 4.3min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718218) |
| stage-b-test-2-npu-a2 (1) | 3.0min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718219) |
| stage-a-unit-test-npu | 4.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718221) |
| multimodal-gen-test-1-npu-a3 | 4.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718230) |
| stage-b-test-16-npu-a3 | 4.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718232) |
| stage-b-test-1-npu-a2 (1) | 3.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718261) |
| stage-b-test-2-npu-a2 (0) | 2.9min | 环境问题 | 下载triton-ascend依赖时自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718278) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718581) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718211

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718216

- **stage-b-test-1-npu-a2 (0)**: 日志显示在构建sglang包后，执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718218

- **stage-b-test-2-npu-a2 (1)**: 日志显示在下载依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），提示联系runner管理员，属于runner环境或容器配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718219

- **stage-a-unit-test-npu**: 作业在运行测试命令时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718221

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718230

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，可能是上游构建未成功上传或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718232

- **stage-b-test-1-npu-a2 (1)**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718261

- **stage-b-test-2-npu-a2 (0)**: 作业在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中，自定义容器实现执行失败，导致作业中断。可能是网络或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718278

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明CI系统尝试下载或访问的远程资源（如模型权重、测试数据或日志文件）在存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594432066/job/91043718581


## [Run #30594369710](https://github.com/sgl-project/sglang/actions/runs/30594369710)
- **分支**: `fix/online-mxfp8-linear`
- **总耗时**: 13.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30594369710

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464495) |
| multimodal-gen-test-2-npu-a3 | 11.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464507) |
| stage-b-test-2-npu-a2 (1) | 12.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464530) |
| stage-b-test-2-npu-a2 (0) | 12.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464532) |
| stage-b-test-1-npu-a2 (1) | 12.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464542) |
| stage-b-test-1-npu-a2 (0) | 12.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464544) |
| stage-b-test-16-npu-a3 | 13.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464552) |
| stage-b-test-4-npu-a3 | 11.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464833) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.9min | 环境问题 | 测试未生成metrics.json，导致性能测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043465545) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示GitHub Actions环境准备、Node.js弃用警告及上传artifact时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464495

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464507

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常（进度74%），但突然报错"Executing the custom container implementation failed"，提示联系runner管理员，属于基础设施/环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464530

- **stage-b-test-2-npu-a2 (0)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464532

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常（HTTP 200），但中途出现 "Executing the custom container implementation failed" 错误，属于自托管 runner 容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464542

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试在捕获批次时（bs=16/12）运行正常，但随后出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464544

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464552

- **stage-b-test-4-npu-a3**: 日志显示在批量捕获测试过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464833

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业运行后未找到/tmp/metrics.json文件，无法上传性能指标，可能因模型推理未完成或环境配置异常导致测试提前结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043465545

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30594369710/job/91043464549) |


## [Run #30593520473](https://github.com/sgl-project/sglang/actions/runs/30593520473)
- **分支**: `main`
- **总耗时**: 19.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30593520473

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 2.7min | 环境问题 | 自定义容器执行失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881029) |
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881044) |
| stage-b-test-4-npu-a3 | 4.6min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881059) |
| stage-b-test-2-npu-a2 (1) | 18.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881073) |
| multimodal-gen-test-2-npu-a3 | 4.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881076) |
| stage-b-test-1-npu-a2 (0) | 18.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881084) |
| stage-b-test-1-npu-a2 (1) | 18.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881098) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.1min | 其他 | 日志被截断，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881373) |

- **stage-b-test-16-npu-a3**: 日志显示在apt-get更新软件包列表后，出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881029

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本警告及上传artifact时未找到diffusion-failures目录，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881044

- **stage-b-test-4-npu-a3**: 作业在加载模型权重时（Multi-thread loading shards 0%）自定义容器实现执行失败，可能是NPU环境或容器配置问题，导致CI提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881059

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881073

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881076

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到91%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881084

- **stage-b-test-1-npu-a2 (1)**: 测试在生成请求完成时（1316/1319）报错“Executing the custom container implementation failed”，属于runner容器执行环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881098

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志仅包含作业初始化和清理阶段，未显示测试执行及失败关键信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881373

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881065) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593520473/job/91040881075) |


## [Run #30593227947](https://github.com/sgl-project/sglang/actions/runs/30593227947)
- **分支**: `main`
- **总耗时**: 6.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30593227947

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 2.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867719) |
| stage-b-test-2-npu-a2 (0) | 5.4min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867734) |
| stage-b-test-1-npu-a2 (0) | 5.6min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867736) |
| stage-b-test-1-npu-a2 (1) | 5.6min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867747) |
| multimodal-gen-test-1-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867752) |
| stage-b-test-4-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867754) |
| multimodal-gen-test-2-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867755) |
| stage-b-test-2-npu-a2 (1) | 5.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867776) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039868103) |

- **stage-b-test-16-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于基础设施/环境配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867719

- **stage-b-test-2-npu-a2 (0)**: 日志显示在初始化torch分布式后，自定义容器实现执行失败，错误为'Executing the custom container implementation failed'，可能是容器环境或配置问题，而非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867734

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重时（25%进度）自定义容器实现执行失败，提示联系自托管runner管理员，可能因NPU环境或容器配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867736

- **stage-b-test-1-npu-a2 (1)**: 日志显示作业在加载模型权重（Multi-thread loading shards 25%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867747

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867752

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或文件被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867754

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867755

- **stage-b-test-2-npu-a2 (1)**: 作业在启动NPU推理服务时，自定义容器实现执行失败，日志显示模型加载和tokenizer初始化正常，但随后容器运行中断，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867776

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败或配置变更，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039868103

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593227947/job/91039867765) |


## [Run #30593023606](https://github.com/sgl-project/sglang/actions/runs/30593023606)
- **分支**: `fuse-swiglu-moe-up-gemm-epilogue`
- **总耗时**: 55.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30593023606

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 44.6min | 其他 | 作业上传了失败产物，但日志未显示具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239012) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-1.zip，表明测试有失败，但未提供失败详情，需查看该产物或完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239012

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239022) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239032) |
| stage-b-test-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239033) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239043) |
| stage-b-test-1-npu-a2 (0) | 42.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239045) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239046) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239047) |
| multimodal-gen-test-1-npu-a3 | 36.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239048) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30593023606/job/91039239249) |


## [Run #30592357194](https://github.com/sgl-project/sglang/actions/runs/30592357194)
- **分支**: `fix/kimi-k2-mxfp4-fp8-per-channel-gfx95`
- **总耗时**: 54.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30592357194

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 44.4min | 其他 | 作业上传了diffusion-failures-npu-2-1工件，表明测试存在失败用例，但日志未显示具体错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303337) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-1工件（5MB），说明multimodal生成测试有失败案例，但未提供具体失败原因，需下载工件分析。
  链接: https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303337

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303308) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303323) |
| stage-b-test-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303326) |
| stage-b-test-1-npu-a2 (0) | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303329) |
| multimodal-gen-test-1-npu-a3 | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303339) |
| stage-b-test-2-npu-a2 (1) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303354) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303376) |
| stage-b-test-2-npu-a2 (0) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303377) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592357194/job/91037303783) |


## [Run #30592200194](https://github.com/sgl-project/sglang/actions/runs/30592200194)
- **分支**: `qiaolin_replayssm`
- **总耗时**: 61.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30592200194

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 46.1min | 精度回归 | 多模态生成测试在NPU上出现diffusion失败，上传了失败工件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801175) |

- **multimodal-gen-test-2-npu-a3**: 作业运行约45分钟后，上传了名为diffusion-failures-npu-2-1的工件，表明diffusion生成测试存在精度或输出不匹配问题，属于精度回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801175

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801080) |
| stage-b-test-16-npu-a3 | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801083) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801086) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801098) |
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801100) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801120) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801126) |
| stage-b-test-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801132) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30592200194/job/91036801279) |


---
*Auto-generated by npu_pr_monitor.py*