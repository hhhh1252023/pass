# NPU CI 执行监控
**生成时间**: 2026-07-31 00:17 UTC
**分析 Run 数**: 26

---

## [Run #30589938453](https://github.com/sgl-project/sglang/actions/runs/30589938453)
- **分支**: `tom/revert-pr10414`
- **总耗时**: 8.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30589938453

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 7.5min | 环境问题 | 自定义容器执行失败，可能是容器环境或配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861319) |
| multimodal-gen-test-1-npu-a3 | 7.4min | 其他 | 日志未显示测试失败的具体原因，仅包含环境警告和工件上传信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861320) |
| stage-b-test-2-npu-a2 (1) | 7.4min | 环境问题 | 自定义容器执行失败，健康检查返回503 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861342) |
| stage-b-test-2-npu-a2 (0) | 7.0min | 环境问题 | 自定义容器执行失败，导致服务不可用 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861346) |
| stage-b-test-16-npu-a3 | 7.4min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861348) |
| multimodal-gen-test-2-npu-a3 | 7.4min | 其他 | 作业日志不完整，未显示测试失败的具体错误信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861353) |
| stage-b-test-1-npu-a2 (1) | 6.8min | 环境问题 | 自定义容器执行失败，健康检查返回503 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861368) |
| stage-b-test-1-npu-a2 (0) | 6.4min | 环境问题 | 健康检查返回503，服务未就绪导致超时 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861371) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.4min | 其他 | 日志未显示测试执行结果，仅包含环境准备和清理步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861631) |

- **stage-b-test-4-npu-a3**: 日志显示在测试运行过程中出现 `Executing the custom container implementation failed` 错误，提示联系自托管运行器管理员，表明容器环境异常导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861319

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅包含Node.js版本弃用警告和工件上传提示（未找到文件）。需要查看更完整的日志才能判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861320

- **stage-b-test-2-npu-a2 (1)**: 服务启动后health_generate接口返回503 Service Unavailable，导致容器执行失败，可能是NPU驱动或模型加载问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861342

- **stage-b-test-2-npu-a2 (0)**: 日志显示服务启动后 health_generate 接口返回 503，随后容器执行失败，提示联系自托管运行器管理员，可能是 NPU 环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861346

- **stage-b-test-16-npu-a3**: 日志显示在加载shards过程中出现错误：'Executing the custom container implementation failed'，提示联系自托管运行器管理员，表明是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861348

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备和清理步骤，缺少测试执行阶段的输出，无法判断失败原因。可能因日志截断或作业在测试前已异常退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861353

- **stage-b-test-1-npu-a2 (1)**: 服务启动后健康检查返回503 Service Unavailable，导致自定义容器实现执行失败，可能是NPU资源不足或配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861368

- **stage-b-test-1-npu-a2 (0)**: NPU服务启动后，/health_generate接口持续返回503，表明模型未完成加载或推理引擎未就绪，最终因超时被自定义容器实现终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861371

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志截断，未包含测试执行和失败信息，仅显示作业初始化、Node.js 版本警告及后处理步骤，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861631

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30589938453/job/91029861296) |


## [Run #30588933414](https://github.com/sgl-project/sglang/actions/runs/30588933414)
- **分支**: `lsyin/verify-buffer-api`
- **总耗时**: 46.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30588933414

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 33.9min | 其他 | 作业日志中未显示明确的失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828633) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件上传成功，无错误或失败步骤。可能失败发生在日志截断部分，或作业因其他未记录原因被标记为失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828633

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828627) |
| stage-b-test-16-npu-a3 | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828632) |
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828634) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828643) |
| stage-b-test-1-npu-a2 (1) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828645) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828646) |
| stage-b-test-1-npu-a2 (0) | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828665) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828694) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588933414/job/91026828870) |


## [Run #30588774955](https://github.com/sgl-project/sglang/actions/runs/30588774955)
- **分支**: `kpham/test-kimi-linear-dcp-dspark`
- **总耗时**: 32.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30588774955

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 28.7min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282205) |
| stage-b-test-4-npu-a3 | 24.3min | 环境问题 | 自定义容器执行失败，可能是容器或运行环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282222) |
| stage-b-test-1-npu-a2 (0) | 31.2min | 环境问题 | 自定义容器执行失败，可能是NPU资源或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282248) |
| multimodal-gen-test-2-npu-a3 | 24.6min | 其他 | 作业日志中未显示测试失败的具体错误，仅包含环境警告和工件上传信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282301) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试用例执行或失败信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282205

- **stage-b-test-4-npu-a3**: 日志显示测试进行到99%时，自定义容器执行失败（Executing the custom container implementation failed），建议检查自托管运行器或容器配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282222

- **stage-b-test-1-npu-a2 (0)**: 在捕获NPU图（bs=56）时，自定义容器实现执行失败，提示联系自托管运行器管理员，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282248

- **multimodal-gen-test-2-npu-a3**: 日志仅包含Node.js版本弃用警告、工件上传成功等信息，未出现测试失败、断言错误或超时等关键错误，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282301

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282212) |
| stage-b-test-16-npu-a3 | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282216) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282217) |
| stage-b-test-2-npu-a2 (1) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282375) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282382) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30588774955/job/91026282791) |


## [Run #30587940948](https://github.com/sgl-project/sglang/actions/runs/30587940948)
- **分支**: `qiaolin_replayssm`
- **总耗时**: 50.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30587940948

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 18.6min | 代码错误 | 测试用例 test_npu_hicache_mla.py 执行失败，返回非零退出码。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794604) |
| multimodal-gen-test-2-npu-a3 | 32.1min | 其他 | 作业日志中未显示明确的失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794637) |

- **stage-b-test-4-npu-a3**: 在 HiCache 测试中，test_npu_hicache_mla.py 运行 414 秒后失败，退出码为 1，导致整体作业失败。其他测试通过，非环境或超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794604

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件已成功上传，但未提供测试执行结果或错误信息，可能因日志截断或作业在后续步骤失败。需查看完整日志以确定具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794637

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794606) |
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794608) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794613) |
| multimodal-gen-test-1-npu-a3 | 34.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794624) |
| stage-b-test-1-npu-a2 (0) | 43.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794630) |
| stage-b-test-1-npu-a2 (1) | 31.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794633) |
| stage-b-test-2-npu-a2 (1) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023794670) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587940948/job/91023795390) |


## [Run #30587825772](https://github.com/sgl-project/sglang/actions/runs/30587825772)
- **分支**: `lsyin/verify-buffer-api`
- **总耗时**: 16.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30587825772

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 2.9min | 其他 | 作业日志不完整，未显示测试执行与失败信息，仅包含环境准备和Node版本警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934857) |
| stage-b-test-1-npu-a2 (0) | 15.5min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934896) |
| stage-b-test-1-npu-a2 (1) | 12.9min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934897) |
| multimodal-gen-test-2-npu-a3 | 15.7min | 环境问题 | 依赖的 blob 文件不存在导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934918) |
| stage-b-test-16-npu-a3 | 6.6min | 环境问题 | 自定义容器执行失败，可能是NPU环境或资源问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934934) |
| stage-b-test-4-npu-a3 | 1.0min | 环境问题 | 自定义容器执行失败，导致作业中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934941) |
| stage-b-test-2-npu-a2 (0) | 12.9min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934953) |
| stage-b-test-2-npu-a2 (1) | 12.9min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934997) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.1min | 其他 | 作业日志不完整，未显示测试执行结果或错误信息，无法判断失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023935265) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、Node 20弃用警告及上传工件步骤，缺少实际测试运行和失败原因，无法判断具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934857

- **stage-b-test-1-npu-a2 (0)**: 日志显示 `Executing the custom container implementation failed`，且运行器使用已弃用的 Node 20，建议联系管理员检查运行器配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934896

- **stage-b-test-1-npu-a2 (1)**: 在运行第二个测试用例时，自定义容器实现执行失败，提示请联系自托管运行器管理员，可能是容器环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934897

- **multimodal-gen-test-2-npu-a3**: 日志显示 Azure Blob 存储中指定的 blob 不存在（BlobNotFound），可能是构建产物或模型文件未正确上传或已被删除，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934918

- **stage-b-test-16-npu-a3**: 日志显示在加载shards过程中出现错误："Executing the custom container implementation failed"，提示联系自托管运行器管理员，表明是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934934

- **stage-b-test-4-npu-a3**: 日志显示在安装依赖包时，自定义容器实现执行失败（Executing the custom container implementation failed），建议联系自托管运行器管理员检查容器配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934941

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行中突然出现"Executing the custom container implementation failed"错误，提示联系自托管运行器管理员，表明是容器或NPU环境异常导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934953

- **stage-b-test-2-npu-a2 (1)**: 作业执行过程中，自定义容器实现失败，错误信息为“Executing the custom container implementation failed”，建议联系自托管运行器管理员排查容器配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023934997

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含环境准备和清理步骤，缺少测试运行阶段的输出，可能因日志截断或作业在测试前已失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023935265

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587825772/job/91023935072) |


## [Run #30587617367](https://github.com/sgl-project/sglang/actions/runs/30587617367)
- **分支**: `rainj-me/rust-server`
- **总耗时**: 43.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30587617367

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 31.2min | 其他 | 作业日志中未显示明确的失败错误，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673956) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件上传成功，无测试失败或异常退出信息，可能因日志截断或作业实际成功但状态误判。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673956

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673906) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673920) |
| multimodal-gen-test-1-npu-a3 | 27.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673947) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673949) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673951) |
| stage-b-test-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673958) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673967) |
| stage-b-test-1-npu-a2 (1) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022673990) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587617367/job/91022674343) |


## [Run #30587552586](https://github.com/sgl-project/sglang/actions/runs/30587552586)
- **分支**: `main`
- **总耗时**: 44.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30587552586

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 30.5min | 其他 | 日志未显示明确失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022533963) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾无错误退出码或失败步骤，仅显示Node 20弃用警告及工件上传成功，可能为作业被手动取消或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022533963

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022533987) |
| stage-b-test-1-npu-a2 (0) | 42.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022533988) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022533990) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022534002) |
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022534005) |
| stage-b-test-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022534014) |
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022534043) |
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022534049) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587552586/job/91022534348) |


## [Run #30587112615](https://github.com/sgl-project/sglang/actions/runs/30587112615)
- **分支**: `qiaolin_replayssm`
- **总耗时**: 14.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30587112615

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 11.4min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137768) |
| multimodal-gen-test-1-npu-a3 | 11.6min | 其他 | 日志未显示测试失败的具体错误，仅包含环境警告和工件上传提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137769) |
| stage-b-test-16-npu-a3 | 13.2min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137774) |
| stage-b-test-1-npu-a2 (1) | 13.1min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137775) |
| stage-b-test-2-npu-a2 (0) | 13.2min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137806) |
| stage-b-test-1-npu-a2 (0) | 11.4min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137812) |
| multimodal-gen-test-2-npu-a3 | 12.4min | 其他 | 日志中未显示测试执行失败的具体错误，仅包含Node.js版本弃用警告和工件上传提示无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137819) |
| stage-b-test-2-npu-a2 (1) | 13.1min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137860) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 13.1min | 其他 | 日志中未显示测试执行失败的具体错误，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021138070) |

- **stage-b-test-4-npu-a3**: 日志显示在测试运行过程中出现 'Executing the custom container implementation failed' 错误，提示联系自托管运行器管理员，表明是运行器环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137768

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败、断言错误或超时等关键信息，仅包含Node.js版本弃用警告和工件未找到的提示，无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137769

- **stage-b-test-16-npu-a3**: 日志显示 'Executing the custom container implementation failed'，表明作业在运行自定义容器时出错，可能是容器配置或环境问题导致，而非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137774

- **stage-b-test-1-npu-a2 (1)**: 作业在运行第二个测试时，自定义容器实现执行失败，提示请联系自托管运行器管理员，导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137775

- **stage-b-test-2-npu-a2 (0)**: 日志显示 `Executing the custom container implementation failed`，提示联系自托管运行器管理员，表明是运行器环境或容器配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137806

- **stage-b-test-1-npu-a2 (0)**: 日志显示 `Executing the custom container implementation failed`，表明 CI 运行环境（K8s 容器）出现异常，导致作业中断，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137812

- **multimodal-gen-test-2-npu-a3**: 作业日志不完整，缺少测试运行阶段的输出，无法判断失败原因。仅看到Node 20弃用警告和`diffusion-failures/`目录无文件被上传，可能测试未运行或日志被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137819

- **stage-b-test-2-npu-a2 (1)**: 日志显示 `Executing the custom container implementation failed`，且存在 Node.js 版本兼容性警告，表明自托管运行器配置或容器环境存在问题，导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137860

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志截断，缺少测试执行阶段的输出，无法判断失败原因。可能因日志不完整或测试未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021138070

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30587112615/job/91021137890) |


## [Run #30586871174](https://github.com/sgl-project/sglang/actions/runs/30586871174)
- **分支**: `main`
- **总耗时**: 11.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30586871174

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 8.8min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289643) |
| stage-b-test-2-npu-a2 (1) | 8.0min | 环境问题 | 自定义容器执行失败，可能是Kubernetes Pod异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289662) |
| stage-b-test-1-npu-a2 (0) | 10.9min | 环境问题 | 自定义容器执行失败，可能是NPU资源或容器配置问题 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289687) |
| multimodal-gen-test-1-npu-a3 | 8.5min | 其他 | 日志未显示测试失败的具体原因，仅包含环境警告和工件上传信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289690) |
| stage-b-test-16-npu-a3 | 10.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289701) |
| stage-b-test-4-npu-a3 | 10.8min | 环境问题 | 自定义容器执行失败，可能因NPU资源或配置问题导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289722) |
| multimodal-gen-test-2-npu-a3 | 8.9min | 其他 | 日志未显示测试失败原因，仅包含Node.js版本弃用警告和工件上传信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289795) |
| stage-b-test-1-npu-a2 (1) | 10.8min | 环境问题 | 自定义容器执行失败，可能是NPU环境或资源问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289805) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.7min | 其他 | 日志未显示测试失败的具体错误，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020290375) |

- **stage-b-test-2-npu-a2 (0)**: 日志显示 `Executing the custom container implementation failed`，提示联系自托管运行器管理员，同时有 Node 20 弃用警告，但核心失败原因是容器执行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289643

- **stage-b-test-2-npu-a2 (1)**: 日志末尾出现'Executing the custom container implementation failed'错误，提示联系自托管Runner管理员，表明K8s容器环境存在问题，而非测试逻辑失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289662

- **stage-b-test-1-npu-a2 (0)**: 日志显示第一个测试通过，但第二个测试开始时容器执行失败，错误信息为'Executing the custom container implementation failed'，属于环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289687

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的详细输出，仅显示Node.js版本弃用警告和工件上传步骤（未找到文件）。需要查看更完整的日志才能判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289690

- **stage-b-test-16-npu-a3**: 日志显示在加载shards过程中出现 `Executing the custom container implementation failed` 错误，提示联系自托管运行器管理员，表明是容器环境问题而非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289701

- **stage-b-test-4-npu-a3**: 日志显示在NPU图捕获阶段出现错误，最终提示自定义容器实现失败，需联系自托管运行器管理员。同时存在Node.js版本弃用警告，但非直接原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289722

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅记录了Node.js 20弃用警告和上传扩散失败工件时未找到文件。需要查看更完整的日志才能判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289795

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行中突然出现 'Executing the custom container implementation failed' 错误，且伴随Node.js版本弃用警告，表明自托管运行器环境异常导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289805

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志截断，缺少测试执行阶段的输出，无法判断失败原因。可能因日志不完整或测试未实际运行导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020290375

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30586871174/job/91020289736) |


## [Run #30585815508](https://github.com/sgl-project/sglang/actions/runs/30585815508)
- **分支**: `feat/grpc-generation-controls`
- **总耗时**: 54.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30585815508

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 39.8min | 其他 | 作业日志中未显示明确的失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843444) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件上传成功，但未提供测试失败或错误的具体信息，可能因日志截断或作业在后续步骤中失败，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843444

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843432) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843453) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843456) |
| stage-b-test-2-npu-a2 (1) | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843461) |
| stage-b-test-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843464) |
| stage-b-test-1-npu-a2 (1) | 28.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843469) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843475) |
| stage-b-test-1-npu-a2 (0) | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843539) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585815508/job/91016843800) |


## [Run #30585730899](https://github.com/sgl-project/sglang/actions/runs/30585730899)
- **分支**: `qiaolin_replayssm`
- **总耗时**: 23.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30585730899

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.9min | 其他 | 作业日志不完整，未显示测试失败或错误信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646044) |
| stage-b-test-1-npu-a2 (1) | 20.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646063) |
| stage-b-test-4-npu-a3 | 21.4min | 环境问题 | NPU显存不足导致容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646098) |
| multimodal-gen-test-1-npu-a3 | 21.5min | 其他 | 作业日志不完整，未显示测试执行和失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646110) |
| stage-b-test-2-npu-a2 (1) | 21.9min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646129) |
| stage-b-test-1-npu-a2 (0) | 20.4min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646227) |

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备、Node版本警告和上传工件步骤，缺少实际测试执行和失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646044

- **stage-b-test-1-npu-a2 (1)**: 日志显示在加载模型权重时出现 `Executing the custom container implementation failed` 错误，提示联系自托管运行器管理员，表明是容器环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646063

- **stage-b-test-4-npu-a3**: 日志显示可用内存仅6.07 GB，而KV Cache分配后内存耗尽，在捕获NPU图时因内存不足导致容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646098

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions环境准备和清理步骤，缺少实际测试命令的输出，无法判断失败原因。可能为作业配置问题或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646110

- **stage-b-test-2-npu-a2 (1)**: 作业执行过程中，自定义容器实现失败，提示联系自托管运行器管理员，可能因容器环境或配置问题导致中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646129

- **stage-b-test-1-npu-a2 (0)**: 日志显示 `Executing the custom container implementation failed`，提示联系自托管运行器管理员，表明是运行器环境或容器配置异常导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646227

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646053) |
| stage-b-test-16-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646075) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646158) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30585730899/job/91016646462) |


## [Run #30584165203](https://github.com/sgl-project/sglang/actions/runs/30584165203)
- **分支**: `rainj-me/rust-server`
- **总耗时**: 53.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30584165203

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 37.8min | 其他 | 作业日志不完整，未显示测试执行与失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671311) |

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备、Node版本警告及上传artifact等步骤，缺少核心测试运行及失败原因，无法判断具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671311

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671252) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671272) |
| stage-b-test-1-npu-a2 (0) | 43.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671275) |
| stage-b-test-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671279) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671289) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671290) |
| stage-b-test-1-npu-a2 (1) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671324) |
| multimodal-gen-test-1-npu-a3 | 35.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671330) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584165203/job/91011671652) |


## [Run #30584051381](https://github.com/sgl-project/sglang/actions/runs/30584051381)
- **分支**: `qiaolin_replayssm`
- **总耗时**: 27.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30584051381

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.4min | 其他 | 作业日志不完整，未显示测试执行与失败信息，仅包含环境准备和Node版本警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212351) |
| stage-b-test-16-npu-a3 | 12.9min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212355) |
| stage-b-test-4-npu-a3 | 6.4min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212388) |
| stage-b-test-1-npu-a2 (0) | 25.5min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212400) |
| multimodal-gen-test-2-npu-a3 | 22.6min | 其他 | 作业日志不完整，未显示测试失败的具体错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212441) |
| stage-b-test-1-npu-a2 (1) | 25.1min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212478) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、Node 20弃用警告及上传工件步骤，未提供任何测试运行、错误堆栈或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212351

- **stage-b-test-16-npu-a3**: 日志显示`Executing the custom container implementation failed`，表明自定义容器执行出错，同时存在Node.js版本警告和算子回退警告，但核心失败原因是容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212355

- **stage-b-test-4-npu-a3**: 作业在运行自定义容器实现时出错，错误信息为“Executing the custom container implementation failed”，建议联系自托管运行器管理员排查容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212388

- **stage-b-test-1-npu-a2 (0)**: 日志显示 `Executing the custom container implementation failed`，表明自托管运行器上的容器执行环境出现问题，导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212400

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备和清理步骤，缺少测试执行阶段的输出，无法判断失败原因。可能因日志截断或作业在测试运行前已失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212441

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试中途出现"Executing the custom container implementation failed"错误，提示联系自托管运行器管理员，表明NPU容器环境异常导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212478

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212363) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212413) |
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212427) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30584051381/job/91011212781) |


## [Run #30583954219](https://github.com/sgl-project/sglang/actions/runs/30583954219)
- **分支**: `improve-ragged-mised-prefill`
- **总耗时**: 44.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30583954219

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 39.2min | 其他 | 作业日志中未包含测试失败的具体错误信息，仅显示Node.js版本弃用警告和工件上传无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870684) |

- **multimodal-gen-test-2-npu-a3**: 日志截断严重，缺少测试执行和失败的关键输出。仅能看到Node 20弃用警告及上传diffusion-failures目录时未找到文件，无法判断实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870684

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870489) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870515) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870518) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870524) |
| multimodal-gen-test-1-npu-a3 | 35.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870538) |
| stage-b-test-16-npu-a3 | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870547) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870557) |
| stage-b-test-1-npu-a2 (0) | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010870584) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583954219/job/91010871110) |


## [Run #30583772463](https://github.com/sgl-project/sglang/actions/runs/30583772463)
- **分支**: `main`
- **总耗时**: 24.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30583772463

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 23.4min | 环境问题 | 自定义容器执行失败，可能是NPU环境或资源问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330149) |
| multimodal-gen-test-1-npu-a3 | 23.5min | 其他 | 作业未显示明确失败，仅存在Node.js版本弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330152) |
| stage-b-test-4-npu-a3 | 23.1min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330171) |
| stage-b-test-1-npu-a2 (0) | 23.8min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330197) |
| multimodal-gen-test-2-npu-a3 | 22.6min | 其他 | 日志未显示测试失败原因，仅包含环境警告和工件上传信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330256) |

- **stage-b-test-1-npu-a2 (1)**: 日志显示模型加载成功，但随后出现容器执行错误，提示联系自托管运行器管理员，表明是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330149

- **multimodal-gen-test-1-npu-a3**: 日志中无测试失败或错误退出信息，仅包含Node 20弃用警告和未找到失败文件的上传提示，作业可能正常完成但未输出关键结果。
  链接: https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330152

- **stage-b-test-4-npu-a3**: 日志显示`Executing the custom container implementation failed`，提示联系自托管运行器管理员，表明容器或NPU环境异常导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330171

- **stage-b-test-1-npu-a2 (0)**: 在运行测试 `test_npu_autoround_moe.py` 时，自定义容器实现执行失败，提示联系自托管运行器管理员，可能因容器环境配置或资源问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330197

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示Node.js版本弃用警告和工件上传步骤（未找到失败文件）。实际失败原因需查看更详细的测试输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330256

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330164) |
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330192) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330212) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330234) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583772463/job/91010330480) |


## [Run #30583541953](https://github.com/sgl-project/sglang/actions/runs/30583541953)
- **分支**: `kpham/test-kimi-linear-dcp-dspark`
- **总耗时**: 44.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30583541953

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 40.4min | 其他 | 作业日志中未显示明确的失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560024) |
| stage-b-test-4-npu-a3 | 18.4min | 代码错误 | 测试用例 test_npu_hicache_mla.py 执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560078) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件上传成功，无错误或失败步骤。可能失败发生在日志截断部分，需查看完整日志以定位具体问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560024

- **stage-b-test-4-npu-a3**: HiCache 测试文件 test_npu_hicache_mla.py 返回退出码 1，导致整体作业失败，5个测试中仅1个通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560078

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560049) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560057) |
| stage-b-test-1-npu-a2 (0) | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560066) |
| multimodal-gen-test-1-npu-a3 | 34.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560072) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560074) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560086) |
| stage-b-test-2-npu-a2 (0) | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560096) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30583541953/job/91009560635) |


## [Run #30582298120](https://github.com/sgl-project/sglang/actions/runs/30582298120)
- **分支**: `main`
- **总耗时**: 20.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30582298120

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 19.6min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367347) |
| stage-b-test-1-npu-a2 (1) | 19.5min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367359) |
| stage-b-test-4-npu-a3 | 17.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367373) |
| multimodal-gen-test-1-npu-a3 | 15.9min | 其他 | 作业日志不完整，未显示测试失败的具体原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367420) |
| multimodal-gen-test-2-npu-a3 | 19.7min | 其他 | 作业日志不完整，未显示测试执行和失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367430) |
| stage-b-test-2-npu-a2 (1) | 15.7min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367567) |

- **stage-b-test-1-npu-a2 (0)**: 在运行第二个测试用例时，自定义容器实现执行失败，提示联系自托管运行器管理员，可能是容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367347

- **stage-b-test-1-npu-a2 (1)**: 日志显示 `Executing the custom container implementation failed`，提示联系自托管运行器管理员，属于环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367359

- **stage-b-test-4-npu-a3**: 在运行第二个测试（test_npu_w4a4_quantization.py）时，自定义容器实现执行失败，提示联系自托管运行器管理员，作业因此终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367373

- **multimodal-gen-test-1-npu-a3**: 日志仅包含环境准备和清理步骤，缺少测试执行阶段的输出，无法判断失败原因。可能因日志截断或作业在测试前已失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367420

- **multimodal-gen-test-2-npu-a3**: 日志仅包含CI环境准备和清理步骤，缺少实际测试命令的输出，无法判断失败原因。可能因日志截断或作业在测试前已异常退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367430

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试任务在接近完成时（1315/1319）突然报错："Executing the custom container implementation failed"，并提示联系自托管运行器管理员，属于运行器环境异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367567

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367331) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367385) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367422) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30582298120/job/91005367708) |


## [Run #30581468022](https://github.com/sgl-project/sglang/actions/runs/30581468022)
- **分支**: `rainj-me/rust-server`
- **总耗时**: 38.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30581468022

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 35.1min | 环境问题 | 自定义容器执行失败，可能是容器环境或资源问题 | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005251) |
| multimodal-gen-test-2-npu-a3 | 29.4min | 其他 | 作业日志中未显示明确失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005278) |
| stage-b-test-1-npu-a2 (0) | 32.2min | 环境问题 | 自定义容器执行失败，可能是容器或运行环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005400) |

- **stage-b-test-4-npu-a3**: 日志显示在捕获批次过程中出现错误，最终提示“Executing the custom container implementation failed”，表明自托管运行器的容器执行环境存在问题，可能与内存或配置有关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005251

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件已成功上传，无错误或异常退出，可能因日志截断或作业实际成功但状态标记有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005278

- **stage-b-test-1-npu-a2 (0)**: 日志显示 `Executing the custom container implementation failed`，提示联系自托管运行器管理员，表明容器执行环境出现问题，而非测试逻辑本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005400

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005247) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005286) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005325) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005326) |
| stage-b-test-2-npu-a2 (0) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005341) |
| multimodal-gen-test-1-npu-a3 | 34.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003005389) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30581468022/job/91003006410) |


## [Run #30578765492](https://github.com/sgl-project/sglang/actions/runs/30578765492)
- **分支**: `flashinfer-pure-allreduce`
- **总耗时**: 44.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30578765492

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 34.1min | 其他 | 作业日志中未显示明确失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573735) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件上传成功，无错误或失败步骤。可能失败发生在日志截断部分，需查看完整日志以定位具体问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573735

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573769) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573777) |
| stage-b-test-16-npu-a3 | 12.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573792) |
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573809) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573812) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573823) |
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573825) |
| stage-b-test-1-npu-a2 (0) | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993573832) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30578765492/job/90993574503) |


## [Run #30574996618](https://github.com/sgl-project/sglang/actions/runs/30574996618)
- **分支**: `optimize-mem-pool-slot-alloc`
- **总耗时**: 43.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30574996618

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 35.2min | 其他 | 日志未显示测试失败的具体原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144577) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败、错误堆栈或超时信息，仅记录了Node.js 20弃用警告及工件上传成功，无法判断失败原因，可能为作业配置或日志截断问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144577

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90980931059) |
| multimodal-gen-test-1-npu-a3 | 37.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144536) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144537) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144540) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144549) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144574) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144611) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144624) |
| stage-b-test-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981144690) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574996618/job/90981145217) |


## [Run #30574497162](https://github.com/sgl-project/sglang/actions/runs/30574497162)
- **分支**: `flashinfer-pure-allreduce`
- **总耗时**: 42.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30574497162

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 35.5min | 其他 | 作业日志不完整，未显示测试执行和失败信息，仅包含环境准备和清理步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979354024) |

- **multimodal-gen-test-2-npu-a3**: 日志截断，缺少核心测试步骤的输出，无法判断失败原因。仅看到Node.js版本弃用警告和工件上传成功，但无测试结果或错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979354024

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979353977) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979353988) |
| stage-b-test-1-npu-a2 (1) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979353992) |
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979353993) |
| stage-b-test-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979353994) |
| stage-b-test-1-npu-a2 (0) | 40.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979354010) |
| stage-b-test-16-npu-a3 | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979354022) |
| multimodal-gen-test-1-npu-a3 | 31.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979354056) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30574497162/job/90979354494) |


## [Run #30572028219](https://github.com/sgl-project/sglang/actions/runs/30572028219)
- **分支**: `mxfp8_nvfp4_mixed_precision_clean_pr`
- **总耗时**: 58.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30572028219

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 36.4min | 其他 | 作业日志不完整，未显示测试失败的具体原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977067) |

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备和清理步骤，缺少测试执行和失败信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977067

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977028) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977029) |
| multimodal-gen-test-1-npu-a3 | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977062) |
| stage-b-test-16-npu-a3 | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977066) |
| stage-b-test-1-npu-a2 (1) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977076) |
| stage-b-test-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977089) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977108) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977113) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30572028219/job/90970977758) |


## [Run #30571122082](https://github.com/sgl-project/sglang/actions/runs/30571122082)
- **分支**: `feat/kimi-linear-pd-dcp-oss`
- **总耗时**: 45.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30571122082

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 33.6min | 其他 | 作业日志中未显示明确失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346889) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件已成功上传，无错误或异常退出，可能因测试结果被截断或失败发生在日志未捕获部分。
  链接: https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346889

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346750) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346753) |
| stage-b-test-1-npu-a2 (1) | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346771) |
| multimodal-gen-test-1-npu-a3 | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346776) |
| stage-b-test-2-npu-a2 (1) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346788) |
| stage-b-test-1-npu-a2 (0) | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346810) |
| stage-b-test-16-npu-a3 | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346824) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021346836) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30571122082/job/91021347210) |


## [Run #30570890438](https://github.com/sgl-project/sglang/actions/runs/30570890438)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 53.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30570890438

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 29.1min | 其他 | 日志未显示测试失败的具体原因，仅包含环境警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161059) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败或错误堆栈，仅包含Node.js版本弃用警告和工件上传成功记录，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161059

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967160999) |
| stage-b-test-2-npu-a2 (0) | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161000) |
| stage-b-test-2-npu-a2 (1) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161077) |
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161082) |
| multimodal-gen-test-1-npu-a3 | 27.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161090) |
| stage-b-test-1-npu-a2 (0) | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161108) |
| stage-b-test-4-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161116) |
| stage-b-test-1-npu-a2 (1) | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161118) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570890438/job/90967161759) |


## [Run #30570425905](https://github.com/sgl-project/sglang/actions/runs/30570425905)
- **分支**: `amd/dsv4-aiter-fused-mhc-cross-layer`
- **总耗时**: 57.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30570425905

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 31.2min | 其他 | 作业日志中未显示明确的失败错误，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613803) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件上传成功，但未提供测试失败或错误的具体信息，可能因日志截断或作业实际成功但状态标记异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613803

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613770) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613788) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613804) |
| stage-b-test-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613815) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613842) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613852) |
| stage-b-test-16-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613860) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965613865) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30570425905/job/90965614398) |


## [Run #30567532375](https://github.com/sgl-project/sglang/actions/runs/30567532375)
- **分支**: `fix-prefill-delayer-slot-delay-bound`
- **总耗时**: 82.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30567532375

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 36.6min | 其他 | 作业日志中未显示明确的失败原因，仅包含 Node.js 版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842712) |
| stage-b-test-1-npu-a2 (0) | 33.6min | 代码错误 | 测试用例 test_npu_autoround_moe.py 执行失败，返回非零退出码。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842736) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件 'diffusion-failures-npu-2-1' 已成功上传，但未提供测试失败或错误的具体信息，可能失败发生在日志截断部分或作业被提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842712

- **stage-b-test-1-npu-a2 (0)**: 在 NPU CI 测试中，quant 目录下的 test_npu_autoround_moe.py 测试失败（exit code 1），导致整体作业失败。其他 3 个测试均通过，问题定位在该测试用例本身。日志未提供具体错误信息，需进一步查看测试输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842736

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842686) |
| multimodal-gen-test-1-npu-a3 | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842697) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842732) |
| stage-b-test-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842741) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842787) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842795) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955842829) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30567532375/job/90955843194) |


---
*Auto-generated by npu_pr_monitor.py*