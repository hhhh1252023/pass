# NPU CI 执行监控
**生成时间**: 2026-08-08 12:27 UTC
**分析 Run 数**: 20

---

## [Run #31255859483](https://github.com/sgl-project/sglang/actions/runs/31255859483)
- **分支**: `codex/mm-cpu-tensor-broadcast`
- **总耗时**: 7.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31255859483

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053197) |
| multimodal-gen-test-2-npu-a3 | 6.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053202) |
| base-b-test-2-npu-a3 / run (0) | 6.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053231) |
| base-b-test-16-npu-a3 / run (0) | 4.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053232) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | 环境问题 | 自定义容器执行失败，apt安装libssl时出错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053259) |
| base-b-test-8-npu-a3 / run (0) | 6.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053264) |
| base-b-test-4-npu-a3 / run (0) | 6.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053282) |
| base-b-test-4-npu-a3 / run (1) | 6.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053285) |
| base-b-test-1-npu-a3 / run (0) | 5.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053287) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.4min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053314) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053315) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.3min | 环境问题 | 自定义容器启动失败，导致作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053334) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 2.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099411734) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053197

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或未产生失败文件，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053202

- **base-b-test-2-npu-a3 / run (0)**: 日志显示服务启动正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053231

- **base-b-test-16-npu-a3 / run (0)**: 作业在运行测试前启动自定义容器时失败，错误为'Executing the custom container implementation failed'，提示联系runner管理员，属于runner或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053232

- **base-a-test-1-npu-a2 / run (0)**: 在安装libssl-dev和升级libssl3时，容器执行失败，报错'Executing the custom container implementation failed'，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053259

- **base-b-test-8-npu-a3 / run (0)**: 日志显示在加载模型分片后，各TP/EP进程获取环境变量时出现异常，随后报错'Executing the custom container implementation failed'，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053264

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在测试运行过程中，自定义容器实现执行失败，提示联系自托管runner管理员。这属于runner环境或容器配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053282

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型权重加载到43%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053285

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU设备或容器环境配置问题，导致无法正常加载权重。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053287

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在安装evalscope依赖构建wheel时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053314

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败，错误信息提示联系自托管runner管理员，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053315

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在执行自定义容器实现时失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053334

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099411734

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255859483/job/93099053322) |


## [Run #31254531302](https://github.com/sgl-project/sglang/actions/runs/31254531302)
- **分支**: `mmangkad/fix-jit-align-single-token-namespace`
- **总耗时**: 45.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31254531302

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 35.0min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942034) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 31.3min | 精度回归 | Qwen3.5-9B GSM8K 精度测试未达基线，准确率0.8低于基线0.835，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942297) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | 性能测试未达到基线，测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93096526024) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败机制触发，因同PR中另一个作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93098295444) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93099542912) |

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py 测试返回退出码1，导致作业失败。该测试耗时368秒，属于功能测试失败，而非环境或超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942034

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试用例 test_npu_qwen3_5_9b_bf16_1p_gsm8k.py 运行1664秒后失败，准确率0.8低于基线0.835，属于精度回归，可能由模型或推理配置变化引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942297

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量397.29，低于基线390.59，未通过性能阈值检查，导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93096526024

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到同PR的base-c-test-perf-8-npu-a3作业失败，将其判定为根因失败，触发了fast-fail机制，导致本作业（16-npu-a3）在启动前被跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93098295444

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在健康检查阶段因其他作业（base-b-test-16-npu-a3、base-c-test-acc-2-npu-a3、base-c-test-perf-8-npu-a3）失败而触发快速失败机制，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93099542912

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942003) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942005) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942035) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942050) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942068) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942074) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942254) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942270) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31254531302/job/93095942272) |


## [Run #31253670254](https://github.com/sgl-project/sglang/actions/runs/31253670254)
- **分支**: `agent/fix-moe-align-single-token-namespace`
- **总耗时**: 8.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31253670254

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 7.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780526) |
| base-b-test-4-npu-a3 / run (1) | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780544) |
| base-b-test-4-npu-a3 / run (0) | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780574) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780855) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.3min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780857) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.0min | 环境问题 | 自定义容器启动失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780871) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780881) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试运行中，但随后出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780526

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存、依赖或上传文件未正确生成，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780544

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780574

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780855

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780857

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在下载依赖包时，执行自定义容器实现失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，与代码或测试无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780871

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780881

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780588) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253670254/job/93093780616) |


## [Run #31253088498](https://github.com/sgl-project/sglang/actions/runs/31253088498)
- **分支**: `main`
- **总耗时**: 53.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31253088498

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431668) |
| base-b-test-1-npu-a3 / run (0) | 5.8min | 精度回归 | HiCache MHA测试精度失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431715) |
| base-b-test-16-npu-a3 / run (0) | 36.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431724) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 代码错误 | HiCache MLA测试用例失败，服务启动或测试断言出错。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431740) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 39.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431828) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.7min | 其他 | 作业实际测试全部通过，失败可能由基础设施或日志收集问题导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431901) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.8min | 性能回归 | 性能测试未达到基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93093191527) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 3.0min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93095982064) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤（未找到文件），未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431668

- **base-b-test-1-npu-a3 / run (0)**: test_npu_hicache_mha.py测试返回exit code 1，测试摘要显示0/11通过，该测试在150秒内失败，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431715

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载Qwen3模型权重后，自定义容器实现执行失败，提示联系self hosted runner管理员，属于NPU测试环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431724

- **base-b-test-4-npu-a3 / run (0)**: 测试test_npu_hicache_mla.py执行失败（exit code 1），涉及DeepSeek-V2-Lite-W8A8模型和HiCache功能，可能因配置参数（如hicache-ratio）或代码逻辑问题导致，需检查具体错误输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431740

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431828

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示2/2测试通过，但出现警告“Failed to copy _temp from pod”，可能因K8s pod清理或临时文件复制失败导致作业被标记为失败，非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431901

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实测吞吐量389.97，低于基线390.5859，未通过性能阈值，测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93093191527

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示在设置CPU亲和性后，执行自定义容器实现时失败，错误提示需联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93095982064

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431642) |
| base-b-test-8-npu-a3 / run (0) | 13.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431713) |
| base-b-test-2-npu-a3 / run (0) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431722) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431725) |
| base-b-test-4-npu-a3 / run (1) | 26.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431735) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431870) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31253088498/job/93092431902) |


## [Run #31252787421](https://github.com/sgl-project/sglang/actions/runs/31252787421)
- **分支**: `main`
- **总耗时**: 8.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31252787421

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640656) |
| multimodal-gen-test-2-npu-a3 | 4.1min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志不完整。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640666) |
| base-b-test-1-npu-a3 / run (0) | 6.0min | 代码错误 | HiCache MHA 测试用例执行失败，导致整个作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640678) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640706) |
| base-b-test-4-npu-a3 / run (1) | 4.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640714) |
| base-b-test-4-npu-a3 / run (0) | 4.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640715) |
| base-b-test-16-npu-a3 / run (0) | 8.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640721) |
| base-b-test-2-npu-a3 / run (0) | 7.6min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640728) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640825) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.2min | 环境问题 | 自定义容器执行失败，导致作业在模型权重加载阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640832) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.5min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640849) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93092075216) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因，可能为作业被提前终止或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640656

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，且未包含实际测试执行或失败信息，可能因前置步骤被截断或作业被提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640666

- **base-b-test-1-npu-a3 / run (0)**: 测试 test_npu_hicache_mha.py 返回退出码 1，0/11 测试通过。可能涉及 HiCache 功能或 MHA 实现问题，需检查相关代码和日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640678

- **base-b-test-8-npu-a3 / run (0)**: 测试本身通过（200/200请求成功），但作业在清理阶段报错“Executing the custom container implementation failed”，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640706

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在加载模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640714

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载模型分片后，执行自定义容器时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640715

- **base-b-test-16-npu-a3 / run (0)**: 测试运行到199/200时，runner报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640721

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型权重加载成功（30.20秒），但随后出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640728

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在启动NPU测试容器时，进程已成功分配CPU亲和性，但随后报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640825

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在加载Qwen3-VL模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640832

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在安装evalscope依赖时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境问题而非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640849

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93092075216

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640719) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31252787421/job/93091640834) |


## [Run #31251999053](https://github.com/sgl-project/sglang/actions/runs/31251999053)
- **分支**: `mmangkad/fix-jit-align-single-token-namespace`
- **总耗时**: 45.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31251999053

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 35.5min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840078) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 30.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840216) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 28.2min | 精度回归 | Qwen3.5-9B GSM8K 测试精度低于基线 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840227) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93092418792) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.8min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93094298961) |

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py返回退出码1，3/6测试通过，该用例耗时368秒，可能因断言失败或运行时错误导致，需检查该测试日志定位具体问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840078

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（吞吐约500 token/s），但中途出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于基础设施环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840216

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 TestNPUQwen3_5_9B_GSM8K 精度为 0.78，低于基线 0.835，导致测试失败，作业退出码为 255。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840227

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量376.82，低于基线390.5859，性能回归约3.5%，未通过性能阈值检查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93092418792

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示执行自定义容器实现时出错，提示联系自托管runner管理员。这属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93094298961

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840077) |
| base-b-test-4-npu-a3 / run (0) | 31.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840085) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840089) |
| base-b-test-4-npu-a3 / run (1) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840090) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840092) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840105) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840243) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251999053/job/93091840244) |


## [Run #31251309361](https://github.com/sgl-project/sglang/actions/runs/31251309361)
- **分支**: `patch-4`
- **总耗时**: 93.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31251309361

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052196) |
| base-b-test-16-npu-a3 / run (0) | 36.1min | 代码错误 | NPU PD分离测试失败，3/6用例通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052241) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.7min | 精度回归 | GLM5 GSM8K 测试精度低于基线，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052394) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.6min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，2/3 通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052436) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.4min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088491569) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最终上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记为失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052196

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，耗时352秒，未显示具体错误信息，但该测试用例失败导致作业整体失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052241

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试 TestNPUGLM5_Top64_Pruned_GSM8K 的 accuracy 为 0.46，低于基线 0.48，未达到精度要求，脚本以退出码 1 结束，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052394

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码 1，其余两个测试通过，表明该模型精度未达预期，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052436

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量387.03，低于基线390.5859，未通过性能阈值检查，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088491569

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052176) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052211) |
| base-b-test-2-npu-a3 / run (0) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052232) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052233) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052243) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052251) |
| base-b-test-4-npu-a3 / run (0) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052262) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052391) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93088052438) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31251309361/job/93091924908) |


## [Run #31250268394](https://github.com/sgl-project/sglang/actions/runs/31250268394)
- **分支**: `cheng/gc-rc-review`
- **总耗时**: 94.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31250268394

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.7min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463788) |
| base-b-test-16-npu-a3 / run (0) | 32.6min | 代码错误 | NPU PD分离测试用例失败，3/6通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463853) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 33.2min | 精度回归 | Qwen3.5-9B GSM8K 测试精度不达标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085464047) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.8min | 性能回归 | 性能测试未达到基线，测试用例失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93086790617) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 29.2min | 性能回归 | 性能测试未达基线，测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93088650871) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463788

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，耗时351秒，其余3个测试通过，表明该测试用例存在功能性问题或环境配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463853

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 TestNPUQwen3_5_9B_GSM8K 实际精度 0.76，低于基线 0.835，导致测试失败，3个测试中0个通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085464047

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量392.85，基线为390.5859，但测试仍失败，可能因延迟或精度等其他指标未达标，需检查完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93086790617

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: TestKimiK25W4A8 吞吐量 2132.3 低于基线 1900.0，但测试仍判定失败，可能因其他指标（如延迟）未达标，导致 0/4 测试通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93088650871

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463783) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463830) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463833) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463918) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463919) |
| base-b-test-4-npu-a3 / run (0) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463935) |
| base-b-test-2-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463936) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 45.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085463979) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085464024) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93085464096) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250268394/job/93089580709) |


## [Run #31250074798](https://github.com/sgl-project/sglang/actions/runs/31250074798)
- **分支**: `pr_test`
- **总耗时**: 15.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31250074798

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972164) |
| multimodal-gen-test-2-npu-a3 | 12.2min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972189) |
| base-b-test-16-npu-a3 / run (0) | 13.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972207) |
| base-b-test-2-npu-a3 / run (0) | 14.1min | 环境问题 | 自定义容器执行失败，NPU后端不支持CUDA设备类型导致SymmetricMemory禁用 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972209) |
| base-b-test-1-npu-a3 / run (0) | 13.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972223) |
| base-b-test-4-npu-a3 / run (0) | 13.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972229) |
| base-b-test-4-npu-a3 / run (1) | 13.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972268) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.9min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972399) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972412) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972453) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93085584753) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到upload-artifact步骤提示diffusion-failures目录无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972164

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972189

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型分片过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972207

- **base-b-test-2-npu-a3 / run (0)**: 日志显示SymmetricMemory不支持cuda设备类型，multimem all-gather被禁用，随后自定义容器实现执行失败，属于NPU环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972209

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972223

- **base-b-test-4-npu-a3 / run (0)**: 日志显示模型权重加载到43%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972229

- **base-b-test-4-npu-a3 / run (1)**: 日志显示torch_npu的transfer_to_npu模块在容器启动时触发警告，随后自定义容器实现执行失败，导致作业在测试开始前终止，属于NPU容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972268

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972399

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但在09:27:31出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972412

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，表明runner在执行自定义容器时失败，属于基础设施/环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972453

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93085584753

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972206) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972269) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31250074798/job/93084972401) |


## [Run #31249996951](https://github.com/sgl-project/sglang/actions/runs/31249996951)
- **分支**: `mm_cache_abstract`
- **总耗时**: 90.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31249996951

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766456) |
| base-b-test-16-npu-a3 / run (0) | 37.1min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766483) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.8min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，导致整体作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766644) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | 性能测试未达到基线，测试用例失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93085601901) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 47.0min | 性能回归 | kimi_k2_6性能测试未通过，2/4用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93087022651) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93088801441) |

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions初始化、Node版本警告及上传diffusion-failures目录（无文件），未包含multimodal-gen测试的实际执行输出或失败原因，可能因日志截断或作业在测试前已结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766456

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py返回退出码1，3/6测试通过，该用例失败是直接原因，其余为环境或依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766483

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 2/3 通过，但 qwen3_5_9b_bf16_1p_gsm8k 测试返回退出码 1，耗时 1413 秒，未达预期精度，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766644

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量391.81低于基线390.5859，未通过性能阈值检查，导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93085601901

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms.py测试返回退出码1，耗时1544秒超过预期，可能因性能未达标或超时导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93087022651

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业被健康检查脚本判定为Fast-fail，原因是其他4个作业（如multimodal-gen-test-2-npu-a3等）失败，导致本作业未实际执行即被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93088801441

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766449) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766457) |
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766477) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766481) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766498) |
| base-b-test-4-npu-a3 / run (0) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766531) |
| base-b-test-2-npu-a3 / run (0) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766540) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766641) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766660) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249996951/job/93084766673) |


## [Run #31249873626](https://github.com/sgl-project/sglang/actions/runs/31249873626)
- **分支**: `mmangkad/fix-trtllm-mla-piecewise-prefill`
- **总耗时**: 78.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31249873626

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465247) |
| base-b-test-16-npu-a3 / run (0) | 34.0min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465311) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.5min | 精度回归 | GLM5 GSM8K 测试精度不达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465427) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 32.2min | 精度回归 | Qwen3.5-9B GSM8K 精度测试未达基线，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465449) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93085084908) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93088822627) |

- **multimodal-gen-test-2-npu-a3**: 作业在运行测试后上传diffusion-failures目录时提示无文件，但日志中间部分被省略，无法判断具体失败点。需查看完整日志确认是测试未执行、全部通过还是其他异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465247

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py执行失败（退出码1），耗时354秒，其余3个测试通过。该测试属于PD分离功能，可能因代码逻辑或环境配置问题导致断言失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465311

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试 test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py 运行1251秒后失败，精度为0.46，低于基线0.48，未达到精度阈值，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465427

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试用例 TestNPUQwen3_5_9B_GSM8K 的 accuracy 为 0.76，低于基线 0.835，精度不达标，测试脚本返回非零退出码，最终作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465449

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-2-npu-a3，触发Fast-fail机制，本作业未实际运行即被跳过，属于CI流程中的级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93085084908

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: PR健康检查发现多个根因失败作业（如multimodal-gen-test-2-npu-a3等），触发fast-fail机制，本作业被跳过，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93088822627

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465271) |
| base-b-test-2-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465274) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465277) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465305) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465314) |
| base-b-test-4-npu-a3 / run (0) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465316) |
| base-b-test-4-npu-a3 / run (1) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465327) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465405) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249873626/job/93084465426) |


## [Run #31249867512](https://github.com/sgl-project/sglang/actions/runs/31249867512)
- **分支**: `main`
- **总耗时**: 123.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31249867512

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 36.4min | 代码错误 | NPU PD分离测试失败，3/6通过，test_npu_pd_disaggregation.py退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367540) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 121.3min | 超时 | NPU精度测试中qwen3_5_9b用例执行超时（3600秒）导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367701) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.9min | 性能回归 | 性能测试未达到基线，测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93086477268) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 3.4min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93088371839) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93089485666) |

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py执行失败（耗时334秒），其余3个测试通过。可能因代码逻辑错误或环境配置问题导致，需查看该测试详细日志定位具体断言或异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367540

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件base-c-test-acc-2-npu-a3中，qwen3_5_9b的GSM8K精度测试运行超过3600秒被判定超时，其余两个用例通过。可能因模型推理性能下降或负载过高导致，需排查该用例的耗时瓶颈。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367701

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms运行1043秒后失败，吞吐量391.01略高于基线390.5859，但测试仍返回退出码1，可能因其他性能指标未达标或断言失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93086477268

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业在“Check PR test health”步骤失败，原因是根因作业base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3失败，触发快速失败机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93088371839

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3，触发fast-fail机制，本作业未实际执行即被跳过，属于依赖的上游作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93089485666

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367561) |
| base-b-test-4-npu-a3 / run (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367569) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367573) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367575) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367583) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367591) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367690) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367714) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249867512/job/93085367721) |


## [Run #31249127808](https://github.com/sgl-project/sglang/actions/runs/31249127808)
- **分支**: `mmangkad/fix-trtllm-mla-piecewise-prefill`
- **总耗时**: 19.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31249127808

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.3min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729650) |
| multimodal-gen-test-2-npu-a3 | 9.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729663) |
| base-b-test-16-npu-a3 / run (0) | 18.2min | 环境问题 | NPU容器内模型权重加载时发生崩溃，导致调度器看门狗超时 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729698) |
| base-b-test-2-npu-a3 / run (0) | 17.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729703) |
| base-b-test-4-npu-a3 / run (0) | 14.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729708) |
| base-b-test-1-npu-a3 / run (0) | 18.1min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729711) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729808) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729811) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 16.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729817) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 11.1min | 超时 | Scheduler watchdog 超时导致容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93083369785) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误，仅有Node.js版本警告和上传artifact时无文件提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729650

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示GitHub Actions环境准备、Node版本警告及上传artifact时未找到文件。无法判断测试是否失败或原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729663

- **base-b-test-16-npu-a3 / run (0)**: 在加载MoE模型权重时，libtorch_python.so中发生崩溃（可能因NPU环境或内存问题），随后调度器看门狗超时，最终自定义容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729698

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型权重加载到50%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是容器资源限制、NPU设备故障或镜像问题导致测试中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729703

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在启动DeepSeek-V2-Lite模型后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729708

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在加载模型权重时（Multi-thread loading shards 0%），自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729711

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在09:07:04出现错误：Executing the custom container implementation failed，提示联系self hosted runner管理员，属于runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729808

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729811

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在09:07:04出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729817

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 Scheduler watchdog timeout (self.watchdog_timeout=300)，TP6 EP6 调度器软超时，随后容器执行失败，作业中止。可能是性能瓶颈或资源竞争导致调度响应过慢。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93083369785

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729688) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729702) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729710) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249127808/job/93082729842) |


## [Run #31249104474](https://github.com/sgl-project/sglang/actions/runs/31249104474)
- **分支**: `startup-weight-load-overlap`
- **总耗时**: 50.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31249104474

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536418) |
| base-b-test-16-npu-a3 / run (0) | 35.0min | 代码错误 | NPU PD分离测试用例失败，3/6通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536515) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.8min | 精度回归 | Qwen3.5-9B GSM8K 测试精度未达基线 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536641) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | 性能测试未达到基线，吞吐量低于预期导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93083038376) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败机制触发，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93084921408) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93086666339) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业最终失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536418

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，耗时375秒，未通过。其他3个测试均通过，表明该测试用例存在功能或逻辑问题，需检查PD分离相关代码。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536515

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 TestNPUQwen3_5_9B_GSM8K 实际精度 0.82，低于基线 0.835，导致测试失败，作业退出码为 255。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536641

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量385.51，低于基线390.5859，未通过性能阈值检查，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93083038376

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现multimodal-gen-test-2-npu-a3作业失败，被判定为根因失败，触发了fast-fail机制，导致本作业未实际运行即被跳过，属于CI流程的连锁跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93084921408

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在“Check PR test health”步骤中检测到其他4个根因作业失败（如multimodal-gen-test-2-npu-a3等），触发fast-fail机制，本作业未实际运行即被取消，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93086666339

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536399) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536401) |
| base-b-test-2-npu-a3 / run (0) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536415) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536463) |
| base-b-test-1-npu-a3 / run (0) | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536465) |
| base-b-test-4-npu-a3 / run (0) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536477) |
| base-b-test-4-npu-a3 / run (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536501) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536536) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536632) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31249104474/job/93082536636) |


## [Run #31248422127](https://github.com/sgl-project/sglang/actions/runs/31248422127)
- **分支**: `codex/diffusion-kernel-cleanup`
- **总耗时**: 33.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31248422127

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 30.8min | 代码错误 | NPU PD分离测试用例失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812939) |
| multimodal-gen-test-2-npu-a3 | 26.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812950) |

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py测试失败，2/6通过，该用例耗时362秒后退出码1，可能涉及PD分离功能逻辑错误或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812939

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤（无文件上传），未出现测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812950

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a3 | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812928) |
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812933) |
| stage-b-test-8-npu-a3 | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812940) |
| stage-b-test-2-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812948) |
| stage-b-test-4-npu-a3 (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812951) |
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812956) |
| stage-b-test-4-npu-a3 (0) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080812962) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080813121) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248422127/job/93080813129) |


## [Run #31248159693](https://github.com/sgl-project/sglang/actions/runs/31248159693)
- **分支**: `codex/kimi-k3-npu-main-20260803`
- **总耗时**: 32.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31248159693

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 4.7min | 代码错误 | NPU采样后端测试失败，测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114638) |
| stage-b-test-2-npu-a3 | 4.9min | 环境问题 | NPU测试启动sglang服务失败，导致测试全部失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114646) |
| stage-b-test-4-npu-a3 (0) | 4.2min | 代码错误 | NPU HiCache MLA 测试失败，测试用例执行报错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114653) |
| multimodal-gen-test-2-npu-a3 | 22.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114658) |
| stage-b-test-16-npu-a3 | 4.5min | 环境问题 | NPU测试用例test_npu_deepep.py启动服务失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114659) |
| stage-b-test-4-npu-a3 (1) | 4.1min | 环境问题 | NPU测试用例test_npu_llada2_mini.py执行失败，0个测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114662) |
| stage-b-test-8-npu-a3 | 4.9min | 代码错误 | NPU测试用例test_npu_eplb_min_rebalancing_utilization_threshold.py执行失败，退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114663) |
| multimodal-gen-test-1-npu-a3 | 29.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114672) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 4.6min | 环境问题 | 作业在启动后立即被清理，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114800) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.8min | 其他 | 日志被截断，未显示测试执行结果，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114807) |

- **stage-b-test-1-npu-a3**: test_npu_sampling_backend.py测试失败，11个测试中仅1个通过，该测试涉及NPU采样后端功能，可能因代码变更或环境配置问题导致功能异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114638

- **stage-b-test-2-npu-a3**: 测试test_npu_tp2_fia_bf16.py在启动sglang serve命令时失败（exit code 255），服务未能成功启动，导致0/6测试全部失败。可能是NPU环境配置或模型加载问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114646

- **stage-b-test-4-npu-a3 (0)**: test_npu_hicache_mla.py 测试在 NPU A3 上运行失败，返回 exit code 1，测试摘要显示 0/5 通过，具体错误信息未在日志中详细展示，但可判断为测试用例本身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114653

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最终上传diffusion-failures目录时提示无文件，说明测试可能通过或失败原因未记录。需查看完整日志以确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114658

- **stage-b-test-16-npu-a3**: DeepSeek-R1-0528-W8A8模型在NPU上以tp=16,ep=16配置启动sglang服务时失败，测试0/6通过，可能是模型加载或NPU环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114659

- **stage-b-test-4-npu-a3 (1)**: 测试文件test_npu_llada2_mini.py在NPU环境下运行失败，返回退出码1，0/4测试通过。可能是NPU环境配置问题或测试用例本身存在兼容性问题，需检查具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114662

- **stage-b-test-8-npu-a3**: 测试文件在运行时报错，导致0/1测试通过。日志显示测试执行了65秒后失败，可能是测试代码本身存在逻辑错误或环境配置问题，需检查该测试的具体报错信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114663

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114672

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试命令，可能是runner被抢占或作业被取消，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114800

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业初始化和清理阶段，未展示测试运行输出或错误信息，可能因日志截断或作业在早期阶段被终止，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114807

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248159693/job/93080114654) |


## [Run #31248031631](https://github.com/sgl-project/sglang/actions/runs/31248031631)
- **分支**: `codex/diffusion-kernel-cleanup`
- **总耗时**: 11.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31248031631

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 9.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783857) |
| stage-b-test-2-npu-a3 | 6.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783860) |
| stage-b-test-1-npu-a3 | 6.4min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783863) |
| stage-b-test-4-npu-a3 (1) | 9.6min | 环境问题 | 自定义容器执行失败，模型权重加载过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783870) |
| stage-b-test-4-npu-a3 (0) | 6.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783884) |
| multimodal-gen-test-2-npu-a3 | 10.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783888) |
| multimodal-gen-test-1-npu-a3 | 10.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783891) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 10.1min | 环境问题 | 作业在准备阶段即被中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079784058) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.0min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079784059) |

- **stage-b-test-16-npu-a3**: 作业在启动测试容器时失败，错误信息为"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783857

- **stage-b-test-2-npu-a3**: 日志显示在加载模型权重时出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于CI环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783860

- **stage-b-test-1-npu-a3**: 日志显示测试在运行第二个用例时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施/环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783863

- **stage-b-test-4-npu-a3 (1)**: 作业在加载模型权重（4/16分片）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783870

- **stage-b-test-4-npu-a3 (0)**: 作业在运行测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783884

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783888

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的输出，仅有runner初始化、Node版本警告和artifact上传（无文件）。可能因日志截断或作业在测试前被取消，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783891

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在下载actions和设置环境后，于plog备份步骤后直接进入清理阶段，未执行任何测试命令，疑似runner或基础设施异常导致提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079784058

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志在测试执行前中断，未包含性能测试数据或错误信息，无法判断失败原因。可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079784059

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783847) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31248031631/job/93079783862) |


## [Run #31247993595](https://github.com/sgl-project/sglang/actions/runs/31247993595)
- **分支**: `main`
- **总耗时**: 28.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31247993595

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (1) | 27.2min | 其他 | 作业实际测试全部通过，但日志末尾出现警告，可能因环境清理问题导致作业被标记为失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722924) |
| stage-b-test-2-npu-a3 | 26.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722935) |
| stage-b-test-4-npu-a3 (0) | 8.0min | 精度回归 | HiCache MLA测试失败，精度不达标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722942) |
| stage-b-test-1-npu-a3 | 6.0min | 代码错误 | NPU HiCache MHA 测试用例执行失败，测试脚本返回非零退出码。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722945) |
| stage-b-test-16-npu-a3 | 10.2min | 代码错误 | NPU PD disaggregation 测试失败，测试用例执行报错。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722948) |

- **stage-b-test-4-npu-a3 (1)**: 日志显示4个NPU测试全部通过（4/4 passed），无测试失败。但末尾有警告：'Failed to copy _temp from pod'，以及Node 20弃用警告。作业失败可能源于基础设施问题（如pod临时文件复制失败），而非代码或测试本身。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722924

- **stage-b-test-2-npu-a3**: 作业在运行测试过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722935

- **stage-b-test-4-npu-a3 (0)**: test_npu_hicache_mla.py在DeepSeek-V2-Lite-W8A8模型上测试失败，运行257秒后报错，测试摘要显示0/5通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722942

- **stage-b-test-1-npu-a3**: 测试 test_npu_hicache_mha.py 在运行 151 秒后报错，测试套件 0/11 通过，脚本退出码为 1，导致 CI 作业失败。具体错误信息未在日志中详细展示，需进一步查看测试输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722945

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py 测试运行 402 秒后失败，返回退出码 1，测试摘要显示 0/6 通过。具体错误信息未在日志中详细展示，但可判断为测试用例本身执行出错，可能涉及 PD 分离功能逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722948

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722936) |
| stage-b-test-8-npu-a3 | 11.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079722937) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079723071) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31247993595/job/93079723074) |


## [Run #31247811647](https://github.com/sgl-project/sglang/actions/runs/31247811647)
- **分支**: `main`
- **总耗时**: 5.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31247811647

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 3.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273297) |
| stage-b-test-4-npu-a3 (0) | 3.6min | 环境问题 | 自定义容器执行失败，测试未完成即中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273303) |
| stage-b-test-2-npu-a3 | 3.6min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273309) |
| stage-a-unit-test-npu | 3.5min | 环境问题 | 自定义容器执行失败，NPU环境初始化或配置异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273315) |
| stage-b-test-4-npu-a3 (1) | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273317) |
| stage-b-test-16-npu-a3 | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273318) |
| stage-b-test-8-npu-a3 | 3.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273320) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 3.8min | 环境问题 | Azure Blob 存储中指定的模型权重文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273542) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.0min | 其他 | 日志被截断，未显示实际测试执行结果，无法判断失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273547) |

- **stage-b-test-1-npu-a3**: 日志显示在TokenizerManager初始化后，出现错误'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273297

- **stage-b-test-4-npu-a3 (0)**: 作业在运行第一个测试test_npu_hicache_mla.py时，自定义容器实现执行失败，错误提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273303

- **stage-b-test-2-npu-a3**: 作业在启动NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员。测试用例已启用但未实际运行，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273309

- **stage-a-unit-test-npu**: 日志显示CANN自定义算子安装成功，但随后容器实现执行失败，报错提示联系自托管runner管理员，可能因NPU驱动、容器配置或资源问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273315

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273317

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或数据在存储中已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273318

- **stage-b-test-8-npu-a3**: 作业在启动NPU推理服务时，TokenizerManager初始化后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273320

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 glm5_top64_pruned_bf16 模型权重在 Azure Blob 存储中缺失或路径错误，可能是文件未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273542

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志仅包含作业启动、环境准备和清理阶段，未包含测试执行、断言或错误信息。作业在准备阶段后直接进入清理，可能因外部中断或日志截断导致，需查看完整日志以定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247811647/job/93079273547


## [Run #31247678438](https://github.com/sgl-project/sglang/actions/runs/31247678438)
- **分支**: `feature/load-reporter`
- **总耗时**: 28.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31247678438

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 3.4min | 代码错误 | 测试文件缺少主入口导致CI失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346826) |
| stage-b-test-16-npu-a3 | 3.9min | 代码错误 | 测试文件缺少main入口导致测试收集失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346831) |
| stage-a-unit-test-npu | 4.5min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346832) |
| stage-b-test-2-npu-a3 | 3.1min | 代码错误 | 测试文件缺少主入口导致收集失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346835) |
| stage-b-test-1-npu-a3 | 3.1min | 代码错误 | 测试文件缺少主入口导致收集失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346840) |
| stage-b-test-4-npu-a3 (1) | 3.5min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346841) |
| stage-b-test-4-npu-a3 (0) | 3.2min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346842) |
| multimodal-gen-test-2-npu-a3 | 21.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346882) |

- **stage-b-test-8-npu-a3**: test/registered/unit/load_reporter/test_standalone_rpc_lifecycle.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在直接运行时被静默跳过，CI 收集测试时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346826

- **stage-b-test-16-npu-a3**: test/registered/unit/load_reporter/test_lifecycle.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 收集测试时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346831

- **stage-a-unit-test-npu**: test_proto_contract.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 收集测试时抛出 ValueError 异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346832

- **stage-b-test-2-npu-a3**: test_standalone_rpc_lifecycle.py 缺少 `if __name__ == "__main__":` 入口，pytest 风格测试在直接运行时会被静默跳过，CI 收集测试时抛出 ValueError，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346835

- **stage-b-test-1-npu-a3**: test_standalone_rpc_lifecycle.py 缺少 `if __name__ == "__main__":` 入口，pytest 风格测试在直接运行时会被静默跳过，CI 收集测试时抛出 ValueError 导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346840

- **stage-b-test-4-npu-a3 (1)**: test/registered/unit/load_reporter/test_standalone_rpc_lifecycle.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 收集测试时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346841

- **stage-b-test-4-npu-a3 (0)**: test/registered/unit/load_reporter/test_decorator.py缺少`if __name__ == "__main__":`入口，导致pytest风格测试在`python3 file.py -f`下静默跳过，CI收集测试时抛出ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346842

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346882

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082346839) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082347153) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31247678438/job/93082347158) |


---
*Auto-generated by npu_pr_monitor.py*