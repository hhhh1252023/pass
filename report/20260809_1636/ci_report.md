# NPU CI 执行监控
**生成时间**: 2026-08-09 08:36 UTC
**分析 Run 数**: 33

---

## [Run #31275350945](https://github.com/sgl-project/sglang/actions/runs/31275350945)
- **分支**: `dev/dlal/norm-quant-fusion-runtime`
- **总耗时**: 7.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31275350945

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.3min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945042) |
| multimodal-gen-test-2-npu-a3 | 1.5min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志被截断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945055) |
| base-b-test-8-npu-a3 / run (0) | 5.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945064) |
| base-b-test-16-npu-a3 / run (0) | 5.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945066) |
| base-b-test-1-npu-a3 / run (0) | 5.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945078) |
| base-b-test-2-npu-a3 / run (0) | 5.3min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945080) |
| base-b-test-4-npu-a3 / run (0) | 4.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945098) |
| base-b-test-4-npu-a3 / run (1) | 4.4min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945112) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.1min | 其他 | 健康检查中的lint检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945212) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945246) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查中的lint检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945257) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.4min | 环境问题 | 自定义容器启动失败，导致测试未执行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945283) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945042

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures目录，但未找到文件，未看到实际测试执行或失败原因，可能因日志截断或测试未运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945055

- **base-b-test-8-npu-a3 / run (0)**: 日志显示模型加载到50%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945064

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型权重加载过程中，自定义容器实现执行失败，提示联系runner管理员，属于NPU自托管环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945066

- **base-b-test-1-npu-a3 / run (0)**: 测试本身通过（Ran 1 test OK），但在运行第二个测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945078

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在TokenizerManager初始化后，出现“Executing the custom container implementation failed”错误，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945080

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载模型后，TP进程获取环境变量时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945098

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在加载模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945112

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行PR健康检查时，lint检查结论为failure，触发了fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945212

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945246

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动阶段执行PR健康检查时，lint检查结论为failure，触发了fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945257

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在运行测试前，执行自定义容器实现时失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945283

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275350945/job/93147945102) |


## [Run #31275325598](https://github.com/sgl-project/sglang/actions/runs/31275325598)
- **分支**: `leon/reuse-batched-mamba-boundary-mask`
- **总耗时**: 33.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31275325598

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 22.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934479) |
| base-b-test-16-npu-a3 / run (0) | 32.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934540) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 31.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934766) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 30.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934779) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93148397902) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 6.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93150493412) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含multimodal-gen测试的具体执行结果或错误信息，仅显示GitHub Actions环境准备、Node版本警告及上传artifact时未找到文件。实际失败原因需查看完整日志或测试输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934479

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型加载到65%时，自定义容器实现执行失败，提示联系runner管理员，属于NPU CI环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934540

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在执行过程中出现错误："Executing the custom container implementation failed. Please contact your self hosted runner administrator."，表明是自托管runner环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934766

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（吞吐约450 token/s），但容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934779

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93148397902

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93150493412

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934482) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934536) |
| base-b-test-2-npu-a3 / run (0) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934553) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934555) |
| base-b-test-4-npu-a3 / run (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934561) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934569) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934572) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934739) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31275325598/job/93147934742) |


## [Run #31273564764](https://github.com/sgl-project/sglang/actions/runs/31273564764)
- **分支**: `agent/fix-dcp-kv-head-mapping`
- **总耗时**: 41.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31273564764

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472914) |
| base-b-test-16-npu-a3 / run (0) | 31.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472963) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.0min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143473082) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 41.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143473105) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 9.4min | 超时 | TokenizerManager watchdog 超时导致容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93146419346) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472914

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），导致测试提前终止。日志显示模型加载正常进行，但容器环境在加载过程中崩溃，属于NPU测试环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472963

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试在正常解码过程中突然报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143473082

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试服务已成功启动并完成一次生成请求，但随后runner报告自定义容器执行失败，可能是NPU容器环境不稳定或资源限制导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143473105

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 TokenizerManager watchdog timeout (self.watchdog_timeout=300)，服务在启动或运行过程中卡住超过300秒，触发软超时，最终导致自定义容器执行失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93146419346

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472937) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472965) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472969) |
| base-b-test-4-npu-a3 / run (0) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472980) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472989) |
| base-b-test-2-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143472992) |
| base-b-test-4-npu-a3 / run (1) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143473000) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143473061) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93143473101) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31273564764/job/93144816898) |


## [Run #31272084121](https://github.com/sgl-project/sglang/actions/runs/31272084121)
- **分支**: `refactor-mxfp4-sm100-trtllm-moerunner`
- **总耗时**: 52.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31272084121

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 22.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708108) |
| base-b-test-16-npu-a3 / run (0) | 32.7min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708121) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 39.3min | 精度回归 | moonshotai_moonlight_16b_a3b 模型 GSM8K 测试失败，返回退出码 1，导致整体测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708318) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 10.9min | 精度回归 | NPU精度测试用例失败，GLM5模型测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708329) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.2min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93140140008) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93145125089) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708108

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时351秒（远低于超时限制），非超时问题。其他3个测试通过，表明环境正常，问题定位在该测试用例本身，可能涉及PD分离功能逻辑或配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708121

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 1/3 通过，2/3 失败。失败项为 moonshotai_moonlight_16b_a3b 的 bf16 单卡 GSM8K 精度测试，运行 633 秒后退出码为 1，可能因精度未达阈值或运行错误导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708318

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，耗时447秒，未超时，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708329

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量378.07，低于基线390.5859，未通过性能阈值检查，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93140140008

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动前的健康检查阶段检测到多个根因作业（如multimodal-gen-test-2-npu-a3等）已失败，因此主动跳过本作业并报错，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93145125089

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708062) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708120) |
| base-b-test-4-npu-a3 / run (1) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708122) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708133) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708142) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708157) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708201) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708237) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31272084121/job/93139708293) |


## [Run #31271565536](https://github.com/sgl-project/sglang/actions/runs/31271565536)
- **分支**: `fix_glm52_pp`
- **总耗时**: 93.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31271565536

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 15.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360236) |
| base-b-test-16-npu-a3 / run (0) | 35.1min | 代码错误 | NPU PD分离测试用例失败，3/6通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360299) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.9min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，其余两个测试通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360511) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.5min | 精度回归 | GLM5 GSM8K 测试精度低于基线，测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360549) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93139868898) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93143809988) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的输出，仅有runner初始化、Node版本警告及上传artifact（无文件）等常规信息，无法判断具体失败点，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360236

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py 测试失败（退出码1），耗时354秒，其余3个测试通过。可能涉及PD分离功能逻辑或环境配置问题，需查看具体断言日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360299

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码1，而 moonshotai_moonlight_16b 和 glm4_7_flash 均通过，表明该模型存在精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360511

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: TestNPUGLM5_Top64_Pruned_GSM8K 测试精度为 0.47，低于基线 0.48，导致测试返回退出码 1，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360549

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业启动前的健康检查检测到multimodal-gen-test-2-npu-a3和base-c-test-acc-16-npu-a3两个根因作业失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93139868898

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在健康检查阶段因multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3等根因作业失败而被Fast-fail跳过，属于级联失败，非本作业自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93143809988

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360203) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360266) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360296) |
| base-b-test-2-npu-a3 / run (0) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360309) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360313) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360332) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360340) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360530) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31271565536/job/93138360555) |


## [Run #31270241845](https://github.com/sgl-project/sglang/actions/runs/31270241845)
- **分支**: `xinyuan/parser-auto-resolution-order`
- **总耗时**: 105.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31270241845

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873509) |
| base-b-test-16-npu-a3 / run (0) | 35.1min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873579) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 25.7min | 性能回归 | 性能测试未达到基线，测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93136487359) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93138969051) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败被快速失败机制跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93140842547) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 2.0min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93145482648) |

- **multimodal-gen-test-2-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873509

- **base-b-test-16-npu-a3 / run (0)**: 测试套件中3/6通过，但test_npu_pd_disaggregation.py返回退出码1，耗时341秒，未显示具体错误信息，可能涉及PD分离功能逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873579

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms 性能测试吞吐量为395.47，低于基线390.5859，未通过性能阈值检查，导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93136487359

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业未实际运行，而是被健康检查脚本因其他三个根因作业（multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3、base-c-test-perf-8-npu-a3）失败而快速跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93138969051

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业未实际运行，因健康检查发现其他作业（如multimodal-gen-test-2-npu-a3等）失败，触发Fast-fail机制，导致本作业被跳过并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93140842547

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在“Check PR test health”步骤中检测到其他根因作业（multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3、base-c-test-perf-8-npu-a3）失败，触发快速失败机制，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93145482648

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873505) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873535) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873556) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873557) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873570) |
| base-b-test-2-npu-a3 / run (0) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873571) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873607) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873667) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873668) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873677) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31270241845/job/93134873678) |


## [Run #31269960300](https://github.com/sgl-project/sglang/actions/runs/31269960300)
- **分支**: `cheng/gc-rc-review`
- **总耗时**: 24.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31269960300

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 21.2min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172517) |
| multimodal-gen-test-2-npu-a3 | 14.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172518) |
| base-b-test-1-npu-a3 / run (0) | 21.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172561) |
| base-b-test-4-npu-a3 / run (0) | 23.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172596) |
| base-b-test-16-npu-a3 / run (0) | 20.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172600) |
| base-b-test-4-npu-a3 / run (1) | 13.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172602) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172785) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 15.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172789) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172837) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.7min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93135477453) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤。无法判断失败原因，可能是日志截断或作业在测试前已异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172517

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions初始化、Node版本警告及上传diffusion-failures目录（无文件），未包含测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172518

- **base-b-test-1-npu-a3 / run (0)**: 日志显示服务启动成功并完成一次生成请求，但随后报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172561

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试运行中容器突然报错“Executing the custom container implementation failed”，随后进入清理流程，属于runner或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172596

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在NPU测试初始化阶段，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner或容器环境配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172600

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型加载到73%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172602

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172785

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172789

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但中途出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172837

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在下载torch等依赖时，执行自定义容器实现失败，错误为'Executing the custom container implementation failed'，可能是容器镜像或运行环境配置问题，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93135477453

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172556) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172569) |
| base-b-test-2-npu-a3 / run (0) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172571) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31269960300/job/93134172774) |


## [Run #31269920786](https://github.com/sgl-project/sglang/actions/runs/31269920786)
- **分支**: `p5-stage-sync-fix`
- **总耗时**: 32.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31269920786

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 9.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269920786/job/93134064811) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的输出，只有GitHub Actions的初始化、上传artifact（无文件）和清理步骤。无法判断具体失败原因，可能为日志截断或作业在测试前已异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269920786/job/93134064811

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31269920786/job/93134064812) |


## [Run #31269623224](https://github.com/sgl-project/sglang/actions/runs/31269623224)
- **分支**: `p1-flux-ln-bitexact`
- **总耗时**: 31.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31269623224

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269623224/job/93133323345) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269623224/job/93133323345

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31269623224/job/93133323358) |


## [Run #31269082608](https://github.com/sgl-project/sglang/actions/runs/31269082608)
- **分支**: `p1-flux-ln-bitexact`
- **总耗时**: 13.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31269082608

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269082608/job/93131938286) |
| multimodal-gen-test-2-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31269082608/job/93131938312) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269082608/job/93131938286

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31269082608/job/93131938312


## [Run #31268411704](https://github.com/sgl-project/sglang/actions/runs/31268411704)
- **分支**: `elastic-ep-cuda-graph-recapture`
- **总耗时**: 60.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31268411704

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206822) |
| base-a-test-1-npu-a2 / run (0) | 1.9min | 环境问题 | rustup 安装 Rust 时下载超时，导致 CI 失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206855) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 该作业因健康检查发现其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206861) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206876) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206889) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查发现根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206913) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206920) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206921) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 作业因健康检查发现其他根因作业失败而被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206971) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 作业因其他根因任务失败被快速失败跳过，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206978) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206984) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206985) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206822

- **base-a-test-1-npu-a2 / run (0)**: 脚本检测到未安装 cargo，尝试通过 rustup 安装 Rust 1.92，但下载 channel-rust-1.92.toml.sha256 时连接内部缓存服务超时，安装失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206855

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查过滤掉级联失败后，根因是base-a-test-1-npu-a2作业失败，导致本作业被Fast-fail跳过，属于级联失败，非本作业自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206861

- **base-b-test-8-npu-a3 / run (0)**: 本作业未实际运行，因健康检查发现根因作业base-a-test-1-npu-a2失败，触发fast-fail机制跳过本作业，属于依赖失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206876

- **base-b-test-2-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因失败作业base-a-test-1-npu-a2，触发快速失败机制，未执行实际测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206889

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，根因是base-a-test-1-npu-a2作业失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206913

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业base-a-test-1-npu-a2失败，本作业作为级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206920

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业base-a-test-1-npu-a2，本作业因级联失败被过滤，最终因根因作业失败而快速失败退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206921

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是base-a-test-1-npu-a2作业失败，导致本作业被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206971

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败任务为base-a-test-1-npu-a2/run，本作业被标记为级联失败并跳过，实际未执行测试，属于上游任务失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206978

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业base-a-test-1-npu-a2，触发快速失败机制，本作业未实际运行即被取消，属于级联跳过而非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206984

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业因健康检查检测到根因作业 base-a-test-1-npu-a2 / run 失败而快速失败，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206985

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31268411704/job/93130206846) |


## [Run #31267741175](https://github.com/sgl-project/sglang/actions/runs/31267741175)
- **分支**: `startup-weight-load-overlap`
- **总耗时**: 98.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31267741175

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182039) |
| base-b-test-16-npu-a3 / run (0) | 36.3min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182115) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141183009) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 37.8min | 性能回归 | 性能测试用例kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms失败，未达到预期性能指标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141183022) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败机制触发，非本作业自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93151397885) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182039

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py返回退出码1，6个测试中3个通过3个失败，该用例耗时379秒后失败，属于功能测试未通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182115

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例 TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms 实测吞吐量 382.18，低于基线 390.5859，未通过性能阈值，脚本退出码 1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141183009

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该性能测试用例执行1572秒后失败（预期1800秒），返回退出码1，导致整体测试1/4通过。可能是模型性能未达标或测试脚本断言失败，需检查具体性能数据。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141183022

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业因其他根因作业（如multimodal-gen-test-2-npu-a3等）失败而被跳过，属于fast-fail连锁反应，本作业本身未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93151397885

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182362) |
| base-b-test-2-npu-a3 / run (0) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182380) |
| base-b-test-4-npu-a3 / run (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182384) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182441) |
| multimodal-gen-test-1-npu-a3 | 26.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182490) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182533) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182537) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182610) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182779) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182794) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141182814) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267741175/job/93141183024) |


## [Run #31267381445](https://github.com/sgl-project/sglang/actions/runs/31267381445)
- **分支**: `main`
- **总耗时**: 85.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31267381445

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.6min | 其他 | 作业未显示明确失败原因，仅上传工件时未找到文件，可能测试未执行或结果为空。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657606) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 代码错误 | NPU测试用例test_npu_hicache_mla.py执行失败，导致整个作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657633) |
| base-b-test-1-npu-a3 / run (0) | 5.9min | 代码错误 | HiCache MHA 测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657638) |
| base-b-test-16-npu-a3 / run (0) | 42.9min | 代码错误 | NPU PD分离测试用例失败，3/6通过，1个测试返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657711) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.6min | 精度回归 | GSM8K 测试精度低于基线，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657808) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93131982477) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.3min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93133900925) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查检测到其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93135572216) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures目录时无文件，但未看到测试失败或错误信息，可能测试被跳过或未产生失败样本，需进一步查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657606

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test/registered/npu/basic_function/HiCache/test_npu_hicache_mla.py在运行267秒后报错，返回exit code 1，测试摘要显示0/5通过，具体错误信息未在日志中详细展示，但可判断为该测试用例本身存在代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657633

- **base-b-test-1-npu-a3 / run (0)**: 测试 test_npu_hicache_mha.py 在启动 sglang serve 后失败，0/11 测试通过。可能是 HiCache 功能或 MHA 模型在 NPU 上存在兼容性问题，需检查相关代码或配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657638

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时393秒，退出码1，其余3个测试通过，属于该测试用例自身的功能或断言错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657711

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: TestNPUQwen3_5_9B_GSM8K 测试精度为 0.8，低于基线 0.835，未达到精度要求，测试用例返回退出码 1，最终作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657808

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现multimodal-gen-test-2-npu-a3、base-b-test-4-npu-a3/run(0)、base-b-test-1-npu-a3/run(0)三个根因作业失败，触发fast-fail机制，本作业未实际执行即被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93131982477

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到其他根因作业（multimodal-gen-test-2-npu-a3等）失败，触发fast-fail机制，本作业未实际执行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93133900925

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段因健康检查发现多个根因作业（如multimodal-gen-test-2-npu-a3等）失败，触发fast-fail逻辑，本作业被主动终止，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93135572216

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657546) |
| base-b-test-8-npu-a3 / run (0) | 11.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657623) |
| base-b-test-4-npu-a3 / run (1) | 27.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657645) |
| base-b-test-2-npu-a3 / run (0) | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657706) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657718) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657774) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657786) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31267381445/job/93127657807) |


## [Run #31266336039](https://github.com/sgl-project/sglang/actions/runs/31266336039)
- **分支**: `fix-grpc-metrics-servicer-version-msg`
- **总耗时**: 76.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31266336039

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.0min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020243) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020273) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是多模态测试作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020279) |
| base-b-test-16-npu-a3 / run (0) | 38.8min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020299) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020416) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.2min | 其他 | 健康检查快速失败，根因是多模态测试任务失败，本任务被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020426) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 28.4min | 精度回归 | GSM8K测试精度0.82低于基线0.835，导致测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020446) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93129715507) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions的常规准备、上传artifact（无文件）和清理步骤，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020243

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，发现同一次运行中的multimodal-gen-test-2-npu-a3作业失败，触发了Fast-fail机制，本作业被跳过并报错退出，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020273

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-2-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020279

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时360秒，可能因代码逻辑错误或环境配置问题导致，需查看具体日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020299

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-2-npu-a3，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020416

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示本任务在“Check PR test health”步骤失败，原因是另一个任务multimodal-gen-test-2-npu-a3失败被判定为根因，本任务作为级联失败被快速跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020426

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: TestNPUQwen3_5_9B_GSM8K用例精度为0.82，低于设定的基线0.835，未达到精度要求，测试返回退出码1，最终作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020446

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3等根因作业失败，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93129715507

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020255) |
| base-b-test-4-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020270) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020284) |
| base-b-test-1-npu-a3 / run (0) | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020289) |
| base-b-test-2-npu-a3 / run (0) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020294) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31266336039/job/93125020463) |


## [Run #31265478658](https://github.com/sgl-project/sglang/actions/runs/31265478658)
- **分支**: `codex/diffusion-kernel-cleanup`
- **总耗时**: 46.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31265478658

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 29.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122829992) |
| multimodal-gen-test-2-npu-a3 | 26.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830006) |
| base-b-test-16-npu-a3 / run (0) | 32.2min | 代码错误 | NPU PD分离测试用例失败，3/6通过，1个测试返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830017) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 27.3min | 精度回归 | Qwen3.5-9B GSM8K 精度测试未达基线，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830094) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93124073060) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因其他根因作业失败被快速失败机制跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93125215792) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 3.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93127288879) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122829992

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830006

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时345秒，退出码1。其他3个测试通过，表明该测试用例存在代码或配置问题，需检查PD分离功能实现。
  链接: https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830017

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 TestNPUQwen3_5_9B_GSM8K 的 accuracy 为 0.82，低于基线 0.835，未通过精度阈值，最终 0/3 测试通过，作业以退出码 1 结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830094

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量381.49，低于基线390.5859，未通过性能阈值检查，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93124073060

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业在健康检查阶段因检测到其他4个根因作业（如multimodal-gen-test等）失败而触发fast-fail，未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93125215792

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93127288879

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830022) |
| base-b-test-4-npu-a3 / run (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830024) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830027) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830036) |
| base-b-test-2-npu-a3 / run (0) | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830043) |
| base-a-test-1-npu-a2 / run (0) | 8.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830063) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830136) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830180) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31265478658/job/93122830220) |


## [Run #31264662455](https://github.com/sgl-project/sglang/actions/runs/31264662455)
- **分支**: `refactor-mxfp4-sm100-trtllm-moerunner`
- **总耗时**: 85.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31264662455

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 19.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795692) |
| base-b-test-16-npu-a3 / run (0) | 30.8min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试文件返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795733) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.3min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，2/3通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795891) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93121298593) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93123492824) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93125120934) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795692

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（耗时340秒），其余3个测试通过。该测试涉及PD分离功能，可能是代码逻辑或环境配置问题导致断言失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795733

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码1，其他两个测试通过，表明该模型精度未达预期，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795891

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量396.48，低于基线390.5859，未通过性能阈值检查，脚本退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93121298593

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到multimodal-gen-test-2-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，触发了fast-fail机制，本作业未实际执行即被跳过，属于依赖的上游失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93123492824

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在健康检查阶段检测到其他作业（如multimodal-gen-test-2-npu-a3等）失败，触发fast-fail机制被跳过，自身未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93125120934

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795664) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795707) |
| base-b-test-4-npu-a3 / run (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795724) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795753) |
| base-a-test-1-npu-a2 / run (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795760) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795776) |
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795793) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795921) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795939) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264662455/job/93120795982) |


## [Run #31264315539](https://github.com/sgl-project/sglang/actions/runs/31264315539)
- **分支**: `brayden/clean-startup-logs`
- **总耗时**: 125.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31264315539

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.0min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928770) |
| base-b-test-16-npu-a3 / run (0) | 32.6min | 代码错误 | NPU PD分离测试用例失败，3/6通过，1个测试文件返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928832) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.6min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93120965205) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败机制触发，非本作业自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93123101393) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.1min | 其他 | 该作业因其他根因作业失败被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93124869476) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93129117311) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败或错误信息，仅有Node.js版本弃用警告及diffusion-failures目录无文件上传提示，可能为作业提前结束或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928770

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（耗时355秒），其余3个测试通过。该测试涉及PD分离功能，可能是代码逻辑或环境配置问题导致断言失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928832

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量384.61，低于基线390.5859，未通过性能阈值检查，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93120965205

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业因其他根因作业（multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3、base-c-test-perf-8-npu-a3）失败而被跳过，属于CI快速失败策略，非本作业代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93123101393

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业被Fast-fail机制跳过，根因是multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3等作业失败，本作业属于级联失败，非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93124869476

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在“Check PR test health”步骤失败，原因是其他作业（multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3、base-c-test-perf-8-npu-a3）存在根因失败，触发了快速失败机制，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93129117311

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 37.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928767) |
| base-b-test-2-npu-a3 / run (0) | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928806) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928848) |
| base-a-test-1-npu-a2 / run (0) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928849) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928850) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928851) |
| base-b-test-8-npu-a3 / run (0) | 10.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119928866) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 46.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119929002) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119929003) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119929020) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31264315539/job/93119929032) |


## [Run #31263883770](https://github.com/sgl-project/sglang/actions/runs/31263883770)
- **分支**: `feature/load-reporter`
- **总耗时**: 16.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31263883770

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 14.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836576) |
| multimodal-gen-test-1-npu-a3 | 15.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836611) |
| base-b-test-2-npu-a3 / run (0) | 15.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836695) |
| base-b-test-1-npu-a3 / run (0) | 15.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836702) |
| base-b-test-16-npu-a3 / run (0) | 14.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836707) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836735) |
| base-b-test-4-npu-a3 / run (0) | 15.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836743) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 15.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836872) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836908) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 15.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836920) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 10.5min | 超时 | Scheduler watchdog 超时导致容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93119352266) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有GitHub Actions环境准备、Node版本警告及上传artifact时未找到文件的通知。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836576

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行步骤或错误信息，仅显示runner初始化、Node版本警告及artifact上传（无文件）。可能因日志截断或作业在测试前被取消，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836611

- **base-b-test-2-npu-a3 / run (0)**: 日志显示torch_npu的transfer_to_npu模块在容器启动时产生ImportWarning和RuntimeWarning，随后自定义容器实现执行失败，导致作业在初始化阶段终止，属于NPU容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836695

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836702

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载到31%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836707

- **base-b-test-4-npu-a3 / run (1)**: 作业在初始化torch分布式时失败，错误为"Executing the custom container implementation failed"，可能是容器或NPU环境配置问题，导致测试无法正常启动。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836735

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载模型分片后，执行自定义容器时失败，错误提示联系自托管runner管理员，可能是NPU设备或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836743

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836872

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在15:28:33时出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836908

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示模型权重加载完成后，在初始化Ascend环境时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836920

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 Scheduler watchdog timeout (self.watchdog_timeout=300)，调度器在300秒内无响应，触发软超时，随后容器执行失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93119352266

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836672) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836745) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31263883770/job/93118836897) |


## [Run #31261862048](https://github.com/sgl-project/sglang/actions/runs/31261862048)
- **分支**: `brayden/k3-fp32-router-logits`
- **总耗时**: 56.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31261862048

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.9min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传空产物提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139417638) |
| base-b-test-16-npu-a3 / run (0) | 36.5min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139417740) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 52.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139417853) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 48.0min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418573) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418618) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败或错误信息，仅有Node 20弃用警告和diffusion-failures目录无文件上传提示，作业可能因无失败样本而正常结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139417638

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py返回退出码1，3/6测试通过，该用例耗时365秒，可能因断言失败或运行时错误导致，需查看具体日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139417740

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在19:23:52时出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139417853

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动多进程NPU环境时，DP/TP/EP各进程获取ASCEND_OPP_PATH后，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境配置或容器兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418573

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量386.25，低于基线390.5859，未通过性能阈值检查，脚本退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418618

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139417921) |
| multimodal-gen-test-1-npu-a3 | 33.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139417950) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418031) |
| base-b-test-4-npu-a3 / run (0) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418145) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418146) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418158) |
| base-b-test-4-npu-a3 / run (1) | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418254) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418324) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418352) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418475) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261862048/job/93139418637) |


## [Run #31261513000](https://github.com/sgl-project/sglang/actions/runs/31261513000)
- **分支**: `mmangkad/fix-trtllm-mla-piecewise-prefill`
- **总耗时**: 48.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31261513000

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 10.7min | 其他 | 作业未执行实际测试，仅上传空失败目录后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112988956) |
| multimodal-gen-test-1-npu-a3 | 31.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112988962) |
| base-b-test-16-npu-a3 / run (0) | 36.4min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989107) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.1min | 精度回归 | Qwen3.5-9B GSM8K 测试精度低于基线，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989344) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93113812103) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93115784817) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93117790789) |

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未发现测试运行记录，且diffusion-failures目录无文件，作业以正常状态结束，无明确失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112988956

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体错误或失败断言，仅显示Node.js 20弃用警告和diffusion-failures目录无文件上传提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112988962

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py返回退出码1，其余3个测试通过。该测试耗时357秒，可能因断言失败或运行时错误导致，需查看具体日志定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989107

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 TestNPUQwen3_5_9B_GSM8K 的 accuracy 为 0.78，低于基线 0.835，未达到精度要求，测试脚本返回非零退出码，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989344

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实测吞吐量373.95，低于基线390.5859，性能回归约4.3%，未通过性能阈值检查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93113812103

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-2-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93115784817

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在“Check PR test health”步骤失败，原因是同一次运行中其他5个作业（如multimodal-gen-test、base-b-test等）已失败，健康检查判定这些为根因，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93117790789

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989054) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989067) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989079) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989095) |
| base-b-test-4-npu-a3 / run (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989103) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989166) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989291) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989315) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261513000/job/93112989365) |


## [Run #31261170117](https://github.com/sgl-project/sglang/actions/runs/31261170117)
- **分支**: `kda-cp-state-preprocess`
- **总耗时**: 45.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31261170117

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 10.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120292) |
| base-b-test-16-npu-a3 / run (0) | 36.4min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120384) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.6min | 精度回归 | Qwen3.5-9B GSM8K 测试精度低于基线，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120533) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112822793) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93114686176) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93116502841) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。无法判断具体失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120292

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py 测试返回退出码1，3/6测试通过，该用例耗时388秒后失败，其余测试均通过，属于该测试用例本身的代码或逻辑问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120384

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 TestNPUQwen3_5_9B_GSM8K 的 accuracy 为 0.77，低于基线 0.835，精度回归明显，导致 3 个测试全部失败，作业退出码为 255。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120533

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实测吞吐量374.0，低于基线390.5859，未通过性能阈值检查，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112822793

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动前的健康检查阶段检测到multimodal-gen-test-2-npu-a3作业失败，触发fast-fail机制，本作业被跳过未实际执行，属于依赖作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93114686176

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 本作业未实际运行，因PR健康检查检测到其他根因作业（如multimodal-gen-test-2-npu-a3等）失败，触发fast-fail机制，导致本作业被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93116502841

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120270) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120342) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120365) |
| base-b-test-4-npu-a3 / run (1) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120371) |
| base-b-test-4-npu-a3 / run (0) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120372) |
| base-b-test-2-npu-a3 / run (0) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120385) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120399) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120537) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120544) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31261170117/job/93112120568) |


## [Run #31260857845](https://github.com/sgl-project/sglang/actions/runs/31260857845)
- **分支**: `main`
- **总耗时**: 15.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31260857845

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.9min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356201) |
| multimodal-gen-test-2-npu-a3 | 14.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356217) |
| base-b-test-16-npu-a3 / run (0) | 15.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356286) |
| base-b-test-1-npu-a3 / run (0) | 5.9min | 代码错误 | NPU HiCache MHA 测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356313) |
| base-b-test-4-npu-a3 / run (0) | 8.3min | 代码错误 | HiCache MLA测试用例失败，服务启动或测试逻辑出错导致退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356376) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356388) |
| base-b-test-2-npu-a3 / run (0) | 11.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356403) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356495) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356513) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 15.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356569) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 10.8min | 超时 | 性能测试执行超时，容器被强制终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111794478) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败产物，但作业最终失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356201

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356217

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载正常进行，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于NPU CI环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356286

- **base-b-test-1-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mha.py 运行报错，返回退出码 1，导致整个作业失败。具体错误信息未在日志中详细展示，但测试未通过，可能涉及代码逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356313

- **base-b-test-4-npu-a3 / run (0)**: 测试test_npu_hicache_mla.py在启动DeepSeek-V2-Lite-W8A8模型服务后失败，0/4通过，可能因HiCache配置（如hicache-ratio 1.2）或模型兼容性问题，需检查具体报错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356376

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行正常，但在14:15:10时出现"Executing the custom container implementation failed"错误，导致作业提前结束。这属于自托管runner环境问题，可能是容器或NPU设备异常，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356388

- **base-b-test-2-npu-a3 / run (0)**: 测试在运行第二个测试文件时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题，而非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356403

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（HTTP 200），但突然报错"Executing the custom container implementation failed"，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356495

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356513

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在14:15:10出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356569

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示Scheduler watchdog超时（300秒），且Capturing batches进度缓慢（bs从18降至8），最终容器实现执行失败，作业因超时被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111794478

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 11.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356285) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356301) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31260857845/job/93111356508) |


## [Run #31260330136](https://github.com/sgl-project/sglang/actions/runs/31260330136)
- **分支**: `brayden/clean-startup-logs`
- **总耗时**: 37.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31260330136

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101428) |
| base-a-test-1-npu-a2 / run (0) | 1.1min | 环境问题 | K8s Pod 拉取镜像失败，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101442) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是另一个作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101450) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是另一个作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101462) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101463) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101499) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101500) |
| base-b-test-16-npu-a3 / run (0) | 36.6min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101571) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查失败导致级联跳过，根因是base-a-test-1-npu-a2作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101638) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101653) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101671) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.2min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101714) |

- **multimodal-gen-test-2-npu-a3**: 日志截断，缺少测试执行部分。仅见Node 20弃用警告及upload-artifact提示无文件上传，无法判断具体失败原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101428

- **base-a-test-1-npu-a2 / run (0)**: 容器镜像 swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-910b-ubuntu22.04-py3.11 拉取失败（ImagePullBackOff），可能是镜像不存在、凭据无效或网络问题，需检查镜像配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101442

- **base-b-test-1-npu-a3 / run (0)**: 本作业因健康检查检测到根因作业 base-a-test-1-npu-a2 / run (0) 失败而触发 fast-fail 跳过，并非自身测试失败，属于级联取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101450

- **base-b-test-2-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业base-a-test-1-npu-a2失败而触发fast-fail机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101462

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业 base-a-test-1-npu-a2 失败，本作业作为级联失败被过滤并快速失败，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101463

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查发现根因作业base-a-test-1-npu-a2失败，触发快速失败机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101499

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业base-a-test-1-npu-a2 / run (0)，触发Fast-fail机制，本作业未实际运行即被跳过，属于依赖的上游作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101500

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py返回退出码1，其余3个测试通过。该测试耗时435秒，未超时，属于功能测试失败，可能是代码逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101571

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业因PR测试健康检查失败被过滤为级联失败，实际根因是base-a-test-1-npu-a2作业失败，导致后续所有相关作业被跳过，最终容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101638

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业因PR健康检查检测到根因作业base-a-test-1-npu-a2失败而快速失败，属于级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101653

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示health-check检测到base-a-test-1-npu-a2作业失败，将其判定为根因，随后触发fast-fail，导致本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101671

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是base-a-test-1-npu-a2作业失败，本作业作为依赖被跳过，并非自身代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101714

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31260330136/job/93110101423) |


## [Run #31259793534](https://github.com/sgl-project/sglang/actions/runs/31259793534)
- **分支**: `main`
- **总耗时**: 26.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31259793534

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 23.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259793534/job/93108737284) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259793534/job/93108737284

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31259793534/job/93108737272) |


## [Run #31259044906](https://github.com/sgl-project/sglang/actions/runs/31259044906)
- **分支**: `codex/kimi-k3-npu-main-20260803`
- **总耗时**: 24.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31259044906

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 17.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889186) |
| multimodal-gen-test-1-npu-a3 | 21.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889189) |
| base-b-test-1-npu-a3 / run (0) | 4.7min | 代码错误 | HiCache测试用例启动sglang服务失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889226) |
| base-b-test-8-npu-a3 / run (0) | 23.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889229) |
| base-b-test-2-npu-a3 / run (0) | 4.3min | 环境问题 | NPU测试用例启动服务失败，导致测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889238) |
| base-b-test-4-npu-a3 / run (1) | 4.9min | 环境问题 | NPU测试用例test_npu_llada2_mini.py启动sglang服务失败，导致测试全部失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889262) |
| base-b-test-16-npu-a3 / run (0) | 23.0min | 环境问题 | NPU调度器进程在初始化时异常退出（exit code -3），导致服务启动失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889264) |
| base-a-test-1-npu-a2 / run (0) | 1.2min | 环境问题 | NPU测试镜像拉取失败，导致Pod无法启动 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889273) |
| base-b-test-4-npu-a3 / run (0) | 4.4min | 代码错误 | NPU HiCache MLA 测试用例执行失败，返回错误码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889285) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889411) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889444) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.7min | 代码错误 | 测试脚本中shell判断语句语法错误导致CI失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889458) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.0min | 精度回归 | GLM5测试用例执行失败，返回退出码1，导致测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889463) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传artifact时未找到diffusion-failures目录，说明测试可能未产生失败文件，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889186

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时无文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889189

- **base-b-test-1-npu-a3 / run (0)**: test_npu_hicache_mha.py测试中，sglang serve命令启动失败（exit code 255），导致0/11测试全部未通过。可能原因包括模型加载问题、参数配置错误或环境依赖缺失。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889226

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889229

- **base-b-test-2-npu-a3 / run (0)**: 测试test_npu_expert_distribution_recorder_mode.py在启动sglang服务时失败（exit code 1），可能是模型加载或NPU环境配置问题，导致0/6测试全部失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889238

- **base-b-test-4-npu-a3 / run (1)**: 测试脚本在启动sglang serve时失败（exit code 255），可能是模型加载、NPU资源或配置问题，导致0/4测试通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889262

- **base-b-test-16-npu-a3 / run (0)**: Rank 0 scheduler died during initialization，退出码-3，可能是NPU设备或驱动环境问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889264

- **base-a-test-1-npu-a2 / run (0)**: K8s Pod因镜像ImagePullBackOff无法启动，镜像swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-910b-ubuntu22.04-py3.11拉取失败，可能是镜像不存在、凭据无效或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889273

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 在运行中报错，导致测试套件0/5通过。具体错误信息未在日志中详细展示，但可判断为测试用例本身存在问题，可能与HiCache MLA功能相关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889285

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动前的健康检查阶段检测到同批次其他作业（base-b-test-4-npu-a3）失败，触发了快速失败机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889411

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示本作业因健康检查检测到其他根因作业（base-b-test-4-npu-a3 和 base-c-test-acc-4-npu-a3）失败而触发快速失败机制，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889444

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 测试用例test_npu_qwen3_vl_30b_a3b_bf16_2p_gsm8k.py执行失败（exit code 1），随后脚本第82行出现`[: 0\n0: integer expression expected`错误，说明shell条件判断中变量为空或格式不正确，导致脚本以非零码退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889458

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py在74秒内失败，0/1测试通过，属于精度回归问题，可能由模型精度或数据变化引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/31259044906/job/93106889463


## [Run #31258262083](https://github.com/sgl-project/sglang/actions/runs/31258262083)
- **分支**: `jit-fp8-per-tensor-scaled-mm`
- **总耗时**: 47.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31258262083

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.1min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志被截断无法定位真实失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965637) |
| multimodal-gen-test-2-npu-a3 | 21.4min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965638) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中的lint检查失败导致作业快速失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965652) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查中的lint检查失败导致作业快速失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965664) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965680) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中的lint检查失败导致作业快速失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965684) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查中lint检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965813) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.1min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965866) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | PR健康检查中的lint检查失败，导致作业在启动前被快速失败终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965882) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965888) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但提示无文件可上传，随后清理退出。中间关键测试日志被省略，无法判断具体失败环节，可能为测试未运行或产物路径错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965637

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965638

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965652

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965664

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行lint健康检查时失败（conclusion=failure），触发了Fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965680

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965684

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行PR健康检查时，lint检查结论为failure，触发fast-fail机制，作业未进入实际测试即退出，属于前置检查失败而非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965813

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行PR健康检查时，lint检查结论为failure，触发fast-fail机制，作业未进入实际测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965866

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在运行前执行健康检查，检测到lint检查结论为failure，触发Fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965882

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段因lint检查失败（conclusion=failure）触发fast-fail机制，未进入实际测试即退出，属于前置检查问题而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965888

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965642) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965649) |
| base-b-test-4-npu-a3 / run (0) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258262083/job/93104965667) |


## [Run #31258256753](https://github.com/sgl-project/sglang/actions/runs/31258256753)
- **分支**: `brayden/k3-fp32-router-logits`
- **总耗时**: 44.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31258256753

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.1min | 其他 | 作业未显示明确失败原因，日志仅包含正常执行和Node 20弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957080) |
| base-b-test-2-npu-a3 / run (0) | 0.9min | 其他 | 健康检查中的lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957132) |
| base-b-test-16-npu-a3 / run (0) | 34.9min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957160) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.4min | 其他 | PR健康检查中的lint检查失败导致作业快速失败，未进入实际测试阶段。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957285) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | PR健康检查中的lint检查失败，导致作业在启动前被快速失败机制终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93105760376) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93107607412) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93109068844) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败、错误或超时信息，仅显示上传artifact时未找到文件（diffusion-failures/目录为空），以及Node 20弃用警告。作业可能因测试未通过但未记录具体错误而失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957080

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行health-check时，lint检查结论为failure，触发了Fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957132

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py 测试返回退出码1，3/6测试通过，该用例执行378秒后失败，属于功能测试未通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957160

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查结论为failure，触发了fast-fail机制，作业提前终止，未运行NPU测试。需先修复PR的lint问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957285

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在运行前执行健康检查，检测到lint检查结论为failure，触发fast-fail机制，作业未进入实际测试阶段即退出，属于前置检查失败而非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93105760376

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行lint健康检查时失败（conclusion=failure），触发fast-fail机制，作业未进入实际测试即退出，属于前置检查问题而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93107607412

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行健康检查时，检测到PR的lint检查结论为failure，触发fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93109068844

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957084) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957117) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957126) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957129) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957148) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957157) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957307) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957341) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258256753/job/93104957343) |


## [Run #31258252875](https://github.com/sgl-project/sglang/actions/runs/31258252875)
- **分支**: `brayden/clean-startup-logs`
- **总耗时**: 53.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31258252875

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953687) |
| base-b-test-16-npu-a3 / run (0) | 36.1min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试文件返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953757) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 51.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953863) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | 性能测试未达基线，测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93105395512) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93107240554) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败被快速失败（fast-fail）跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93108948240) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953687

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（退出码1），耗时338秒，其余3个测试通过。可能是PD分离功能相关代码或测试用例存在问题，需检查该测试的具体断言和日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953757

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，提示联系runner管理员，属于runner容器环境问题而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953863

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms的吞吐量为399.59，低于基线390.5859，导致测试失败，退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93105395512

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段因健康检查检测到multimodal-gen-test-2-npu-a3作业失败，触发fast-fail机制，本作业被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93107240554

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现其他作业（如multimodal-gen-test-2-npu-a3等）失败，本作业作为级联失败被跳过，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93108948240

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953678) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953712) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953723) |
| base-b-test-1-npu-a3 / run (0) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953728) |
| base-b-test-2-npu-a3 / run (0) | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953729) |
| base-b-test-4-npu-a3 / run (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953730) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953740) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953831) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953857) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31258252875/job/93104953873) |


## [Run #31256483266](https://github.com/sgl-project/sglang/actions/runs/31256483266)
- **分支**: `codex/mm-cpu-tensor-broadcast`
- **总耗时**: 43.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31256483266

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 18.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544370) |
| base-b-test-16-npu-a3 / run (0) | 34.5min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544465) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 28.3min | 精度回归 | Qwen3.5-9B GSM8K 测试精度不达标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544601) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.1min | 性能回归 | NPU性能测试未通过，吞吐量未达到基线要求。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93101055834) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.2min | 其他 | 健康检查快速失败机制触发，非本作业自身失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93103958719) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93104292611) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544370

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时347秒，退出码1。其他3个测试通过，非环境或超时问题，属于该测试用例本身的代码或逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544465

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 TestNPUQwen3_5_9B_GSM8K 实际精度 0.76，低于基线 0.835，导致测试失败，作业退出码为 255。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544601

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms失败，实际吞吐量390.5低于基线390.5859，导致测试退出码为1，整体作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93101055834

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业被健康检查脚本判定为根因失败（multimodal-gen-test-2-npu-a3等），触发fast-fail跳过，实际是其他作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93103958719

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业未实际运行，而是因PR中其他作业（multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3、base-c-test-perf-8-npu-a3）失败触发了Fast-fail机制，属于级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93104292611

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544360) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544412) |
| base-b-test-4-npu-a3 / run (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544414) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544442) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544457) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544459) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544463) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544592) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544619) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256483266/job/93100544647) |


## [Run #31256436512](https://github.com/sgl-project/sglang/actions/runs/31256436512)
- **分支**: `fix/faster-fp32-lm-head-mm`
- **总耗时**: 173.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31256436512

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389046) |
| base-b-test-16-npu-a3 / run (0) | 36.5min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389087) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.6min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100778444) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 25.1min | 性能回归 | 性能测试未达到基线，测试用例失败导致作业退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93102835614) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures工件时未找到文件，可能测试未运行或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389046

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时334秒，退出码1，其余3个测试通过。可能是PD分离功能相关代码或测试用例存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389087

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量377.64，低于基线390.5859，性能回归约3.3%，导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100778444

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试用例test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行失败，吞吐量6214.38低于基线6189.0，未通过性能验证，4个测试全部失败，脚本退出码255。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93102835614

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389059) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389062) |
| base-b-test-1-npu-a3 / run (0) | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389070) |
| multimodal-gen-test-1-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389072) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389085) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389103) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389104) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389188) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389464) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389486) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93100389488) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93104551589) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 86.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31256436512/job/93108534175) |


## [Run #31256268250](https://github.com/sgl-project/sglang/actions/runs/31256268250)
- **分支**: `codex/mm-cpu-tensor-broadcast`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31256268250

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104520) |
| multimodal-gen-test-2-npu-a3 | 4.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104525) |
| base-a-test-1-npu-a2 / run (0) | 3.9min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104534) |
| base-b-test-16-npu-a3 / run (0) | 3.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104548) |
| base-b-test-2-npu-a3 / run (0) | 4.1min | 环境问题 | 自定义容器执行失败，导致测试未开始即终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104555) |
| base-b-test-4-npu-a3 / run (0) | 4.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104559) |
| base-b-test-4-npu-a3 / run (1) | 4.1min | 环境问题 | 自定义容器执行失败，导致测试未开始即中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104568) |
| base-b-test-1-npu-a3 / run (0) | 4.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104597) |
| base-b-test-8-npu-a3 / run (0) | 3.8min | 环境问题 | 自定义容器执行失败，NPU测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104605) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | 其他 | 测试套件未找到任何测试用例，导致脚本判断失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104657) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.8min | 环境问题 | 自定义容器执行失败，导致作业在启动阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104659) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 0.7min | 环境问题 | 自定义容器启动失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104674) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.1min | 环境问题 | 自定义容器启动失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104677) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、Node版本警告及上传diffusion-failures目录（无文件）的步骤，未显示任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104520

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node 20弃用警告和上传artifact时未找到失败文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104525

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行测试前执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境配置或容器启动问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104534

- **base-b-test-16-npu-a3 / run (0)**: 作业在运行测试前，执行自定义容器实现时失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104548

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试已启用并开始执行，但随后报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境配置或运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104555

- **base-b-test-4-npu-a3 / run (0)**: 作业在运行test_npu_hicache_mla.py时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104559

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在测试启动后立即出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104568

- **base-b-test-1-npu-a3 / run (0)**: 作业在运行test_npu_hicache_mha.py时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104597

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试命令已开始执行，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员。这属于runner环境或容器配置问题，导致测试无法在NPU上运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104605

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示“No tests found for hw=NPU, suite=base-c-test-acc-8-npu-a3”，测试数量为0，随后shell脚本因整数表达式错误退出，属于配置或迁移问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104657

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在设置环境变量后，执行自定义容器时出现错误“Executing the custom container implementation failed”，可能是容器镜像或运行时环境问题，需联系自托管 runner 管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104659

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动自定义容器实现时失败，错误提示需联系自托管runner管理员，属于NPU CI环境配置或容器运行时问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104674

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在设置环境变量后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，非测试代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31256268250/job/93100104677


## [Run #31255963135](https://github.com/sgl-project/sglang/actions/runs/31255963135)
- **分支**: `leon/reuse-batched-mamba-boundary-mask`
- **总耗时**: 163.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31255963135

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266012) |
| base-b-test-16-npu-a3 / run (0) | 34.1min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266020) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.2min | 性能回归 | 性能测试未达到基线，测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099695993) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 46.9min | 性能回归 | kimi_k2_6性能测试未达预期，测试失败导致作业退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93102094288) |

- **multimodal-gen-test-2-npu-a3**: 日志显示作业启动后直接进入上传工件阶段，且提示未找到diffusion-failures目录，说明测试可能提前结束或未产生失败文件，但具体失败原因未在日志中体现。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266012

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py 测试返回退出码1，导致作业失败。该测试属于pd_disaggregation功能，可能涉及代码逻辑或环境配置问题，需进一步查看具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266020

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms 实测吞吐397.56，低于基线390.5859，未通过性能阈值，导致测试脚本退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099695993

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms.py测试失败（exit code 1），耗时1563秒，未通过性能指标，其余2个测试通过，整体作业因该测试失败而终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93102094288

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266013) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266042) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266047) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266093) |
| base-b-test-4-npu-a3 / run (1) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266098) |
| base-b-test-2-npu-a3 / run (0) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266112) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266143) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266248) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266259) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 43.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266271) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93099266297) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93103773151) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 74.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255963135/job/93107614935) |


## [Run #31255936813](https://github.com/sgl-project/sglang/actions/runs/31255936813)
- **分支**: `optimize-mem-pool-slot-alloc`
- **总耗时**: 171.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31255936813

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195341) |
| base-b-test-16-npu-a3 / run (0) | 37.6min | 代码错误 | NPU PD分离测试用例失败，3/6通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195403) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.6min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099639551) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 40.8min | 性能回归 | NPU性能测试中kimi_k2_6用例失败，未达到预期性能指标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93101361625) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件，无法判断具体失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195341

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，耗时340秒，可能因PD分离功能实现或配置问题导致测试断言失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195403

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实测吞吐量381.69，低于基线390.5859，未通过性能阈值检查，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099639551

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms用例返回退出码1，耗时1684秒超过预估1800秒，疑似性能未达标或执行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93101361625

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195355) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195399) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195427) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195432) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195455) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195459) |
| base-b-test-4-npu-a3 / run (0) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195463) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195576) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195602) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195631) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93099195650) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93103140902) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 79.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31255936813/job/93107848745) |


---
*Auto-generated by npu_pr_monitor.py*