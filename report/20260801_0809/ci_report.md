# NPU CI 执行监控
**生成时间**: 2026-08-01 00:09 UTC
**分析 Run 数**: 17

---

## [Run #30671121844](https://github.com/sgl-project/sglang/actions/runs/30671121844)
- **分支**: `DSV4_MXFP4_MTP`
- **总耗时**: 66.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30671121844

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 27.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944811) |

- **multimodal-gen-test-2-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944811

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944790) |
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944797) |
| stage-b-test-1-npu-a2 (0) | 41.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944799) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944812) |
| multimodal-gen-test-1-npu-a3 | 38.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944813) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944814) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944820) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944895) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30671121844/job/91288944932) |


## [Run #30670957752](https://github.com/sgl-project/sglang/actions/runs/30670957752)
- **分支**: `hicache-dcp-l2`
- **总耗时**: 55.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30670957752

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438320) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行结果或错误信息，仅有Node.js 20弃用警告和diffusion-failures目录无文件上传提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438320

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438279) |
| multimodal-gen-test-1-npu-a3 | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438308) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438314) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438316) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438324) |
| stage-b-test-4-npu-a3 | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438325) |
| stage-b-test-1-npu-a2 (0) | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438332) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438335) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670957752/job/91288438835) |


## [Run #30670956905](https://github.com/sgl-project/sglang/actions/runs/30670956905)
- **分支**: `cheng/inkling-drop-layer-cache-by-id`
- **总耗时**: 64.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30670956905

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500877) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。结尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500877

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500866) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500881) |
| stage-b-test-1-npu-a2 (1) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500887) |
| multimodal-gen-test-1-npu-a3 | 33.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500888) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500889) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500905) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500915) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288500945) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670956905/job/91288501247) |


## [Run #30670852421](https://github.com/sgl-project/sglang/actions/runs/30670852421)
- **分支**: `dev/dlal/norm-quant-fusion`
- **总耗时**: 27.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30670852421

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152298) |
| stage-b-test-4-npu-a3 | 25.9min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152305) |
| stage-b-test-1-npu-a2 (0) | 25.8min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152319) |
| multimodal-gen-test-2-npu-a3 | 25.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152323) |
| stage-b-test-1-npu-a2 (1) | 24.0min | 超时 | TokenizerManager watchdog超时导致服务启动失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152363) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记为失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152298

- **stage-b-test-4-npu-a3**: 作业在加载模型权重时（Multi-thread loading shards 0%）自定义容器执行失败，可能是NPU环境或容器资源问题导致，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152305

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重（50%进度）时，自定义容器实现执行失败，导致作业终止。可能是容器环境不稳定或资源限制，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152319

- **multimodal-gen-test-2-npu-a3**: 日志被截断，缺少关键测试输出。仅看到Node 20弃用警告和diffusion-failures目录无文件上传提示，无法判断具体失败原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152323

- **stage-b-test-1-npu-a2 (1)**: 日志显示TokenizerManager watchdog timeout (self.watchdog_timeout=300)，服务在启动过程中卡住超过300秒未响应，最终被watchdog终止，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152363

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152296) |
| stage-b-test-16-npu-a3 | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152301) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152321) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152325) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670852421/job/91288152732) |


## [Run #30670522773](https://github.com/sgl-project/sglang/actions/runs/30670522773)
- **分支**: `main`
- **总耗时**: 42.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30670522773

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 30.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174037) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败的具体错误信息，仅显示Node 20弃用警告和diffusion-failures目录无文件上传提示，无法判断真实失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174037

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174017) |
| stage-b-test-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174028) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174040) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174046) |
| stage-b-test-1-npu-a2 (1) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174064) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174086) |
| stage-b-test-1-npu-a2 (0) | 41.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174087) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174095) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670522773/job/91287174429) |


## [Run #30670313904](https://github.com/sgl-project/sglang/actions/runs/30670313904)
- **分支**: `cctry-triton-load-watch`
- **总耗时**: 46.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30670313904

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 30.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546803) |

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本警告及上传artifact时未找到文件，未包含multimodal测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546803

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546769) |
| stage-b-test-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546775) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546779) |
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546807) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546812) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546814) |
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546824) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286546843) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30670313904/job/91286547160) |


## [Run #30669840502](https://github.com/sgl-project/sglang/actions/runs/30669840502)
- **分支**: `kpham/test-kimi-linear-dcp-dspark`
- **总耗时**: 45.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30669840502

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 28.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097500) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分省略，末尾显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败产物或提前退出，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097500

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097447) |
| stage-b-test-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097450) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097458) |
| stage-b-test-1-npu-a2 (1) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097471) |
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097478) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097492) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097502) |
| stage-b-test-2-npu-a2 (1) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097513) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669840502/job/91285097881) |


## [Run #30669783464](https://github.com/sgl-project/sglang/actions/runs/30669783464)
- **分支**: `cheng/inkling-drop-layer-cache-by-id`
- **总耗时**: 19.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30669783464

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 19.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408119) |
| stage-b-test-1-npu-a2 (0) | 19.0min | 环境问题 | NPU测试执行过程中自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408129) |
| stage-b-test-4-npu-a3 | 17.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408131) |
| stage-b-test-1-npu-a2 (1) | 19.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408140) |
| multimodal-gen-test-2-npu-a3 | 16.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408141) |
| stage-b-test-2-npu-a2 (1) | 18.8min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408175) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未产生失败样本或提前退出，需查看完整日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408119

- **stage-b-test-1-npu-a2 (0)**: 第一个测试通过后，第二个测试test_npu_piecewise_graph_prefill.py启动时，自定义容器执行失败（Executing the custom container implementation failed），可能是NPU资源或容器环境问题，非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408129

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常，但在进度约97%时，GitHub Actions 报错“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408131

- **stage-b-test-1-npu-a2 (1)**: 日志显示在模型初始化后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408140

- **multimodal-gen-test-2-npu-a3**: 日志截断，缺少核心测试执行部分。仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408141

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常（Prefill batch正常处理），但随后出现"Executing the custom container implementation failed"错误，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408175

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408142) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408149) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408166) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669783464/job/91285408506) |


## [Run #30669511141](https://github.com/sgl-project/sglang/actions/runs/30669511141)
- **分支**: `ajtulloch/kv-vmm-no-implicit-headers`
- **总耗时**: 48.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30669511141

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905085) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误或失败断言，仅有Node.js 20弃用警告和diffusion-failures目录无文件上传提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905085

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905035) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905038) |
| stage-b-test-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905050) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905057) |
| multimodal-gen-test-1-npu-a3 | 27.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905062) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905071) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905073) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905097) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669511141/job/91288905377) |


## [Run #30669370269](https://github.com/sgl-project/sglang/actions/runs/30669370269)
- **分支**: `optimize-mem-pool-slot-alloc`
- **总耗时**: 43.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30669370269

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633124) |

- **multimodal-gen-test-2-npu-a3**: 日志被截断，缺少关键测试输出。仅看到Node 20弃用警告和diffusion-failures目录无文件上传，无法判断具体失败原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633124

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633079) |
| stage-b-test-16-npu-a3 | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633094) |
| stage-b-test-1-npu-a2 (0) | 41.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633102) |
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633106) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633110) |
| stage-b-test-1-npu-a2 (1) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633111) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633112) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633122) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30669370269/job/91283633479) |


## [Run #30668891571](https://github.com/sgl-project/sglang/actions/runs/30668891571)
- **分支**: `main`
- **总耗时**: 30.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30668891571

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 29.8min | 超时 | Scheduler watchdog 超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237168) |
| stage-b-test-1-npu-a2 (1) | 20.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237171) |
| multimodal-gen-test-1-npu-a3 | 29.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237179) |
| stage-b-test-4-npu-a3 | 29.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237181) |
| multimodal-gen-test-2-npu-a3 | 24.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237224) |

- **stage-b-test-1-npu-a2 (0)**: 日志显示 Scheduler watchdog timeout (self.watchdog_timeout=300)，调度器在300秒内无响应，触发软超时，最终导致自定义容器执行失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237168

- **stage-b-test-1-npu-a2 (1)**: 日志显示模型权重加载完成后，自定义容器实现执行失败，提示联系自托管 runner 管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237171

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node 20弃用警告和上传artifact时无文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237179

- **stage-b-test-4-npu-a3**: 作业在启动TokenizerManager后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237181

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237224

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237163) |
| stage-b-test-2-npu-a2 (1) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237175) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237196) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237204) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668891571/job/91282237515) |


## [Run #30668443708](https://github.com/sgl-project/sglang/actions/runs/30668443708)
- **分支**: `feat/capture_graph`
- **总耗时**: 42.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30668443708

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 24.5min | 代码错误 | HiCache MLA 测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928018) |
| multimodal-gen-test-2-npu-a3 | 27.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928079) |

- **stage-b-test-4-npu-a3**: test_npu_hicache_mla.py 测试用例执行失败（exit code 1），其余4个测试中2个通过2个失败，整体测试通过率2/5，该测试用例本身存在代码或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928018

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有GitHub Actions环境准备、Node版本警告及上传artifact提示无文件。无法判断具体失败原因，可能为日志截断或作业在测试前被中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928079

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280927984) |
| stage-b-test-1-npu-a2 (1) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928029) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928048) |
| stage-b-test-1-npu-a2 (0) | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928050) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928056) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928060) |
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928085) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668443708/job/91280928360) |


## [Run #30668122218](https://github.com/sgl-project/sglang/actions/runs/30668122218)
- **分支**: `yusheng/lora-dpattn-attn-tp`
- **总耗时**: 44.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30668122218

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852163) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误或失败断言，仅显示Node.js 20弃用警告和上传工件时未找到文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852163

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 13.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852164) |
| stage-b-test-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852179) |
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852180) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852184) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852191) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852204) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852208) |
| stage-b-test-1-npu-a2 (0) | 41.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852222) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668122218/job/91279852424) |


## [Run #30668069293](https://github.com/sgl-project/sglang/actions/runs/30668069293)
- **分支**: `feat/capture_graph`
- **总耗时**: 7.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30668069293

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 6.0min | 环境问题 | 自定义容器执行失败，服务健康检查返回503 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711759) |
| multimodal-gen-test-1-npu-a3 | 6.0min | 其他 | 日志不完整，未显示测试失败的具体原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711778) |
| stage-b-test-2-npu-a2 (0) | 5.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711786) |
| stage-b-test-4-npu-a3 | 2.5min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711794) |
| stage-b-test-1-npu-a2 (1) | 5.9min | 环境问题 | NPU测试环境健康检查失败，服务启动后返回503。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711803) |
| stage-b-test-16-npu-a3 | 6.0min | 环境问题 | 自定义容器执行失败，模型分片加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711806) |
| multimodal-gen-test-2-npu-a3 | 6.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711836) |
| stage-b-test-2-npu-a2 (1) | 5.8min | 环境问题 | 自定义容器执行失败，模型加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711838) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 2.5min | 其他 | 作业日志被截断，未显示实际失败原因，仅见清理和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279712036) |

- **stage-b-test-1-npu-a2 (0)**: NPU服务启动后/health_generate接口持续返回503，随后自定义容器实现执行失败，可能是容器环境或NPU资源问题导致服务未就绪。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711759

- **multimodal-gen-test-1-npu-a3**: 作业在运行测试后上传diffusion-failures目录时提示无文件，但日志中未包含测试执行的具体输出或错误信息，无法判断是测试通过还是失败原因被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711778

- **stage-b-test-2-npu-a2 (0)**: 作业在启动阶段执行自定义容器时失败，日志显示torch分布式初始化后出现容器执行错误，可能是NPU资源或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711786

- **stage-b-test-4-npu-a3**: 日志显示在安装Python依赖时，缓存服务返回了无效的networkx包格式，随后自定义容器实现执行失败，作业提前终止，属于运行环境或缓存服务异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711794

- **stage-b-test-1-npu-a2 (1)**: 服务启动后/health_generate返回503，随后自定义容器执行失败，可能是NPU资源或环境配置问题导致服务未就绪。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711803

- **stage-b-test-16-npu-a3**: 作业在加载模型分片（约35%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711806

- **multimodal-gen-test-2-npu-a3**: 日志中仅包含Node.js版本弃用警告、上传diffusion-failures目录时未找到文件等常规信息，未出现测试失败的具体错误或退出码，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711836

- **stage-b-test-2-npu-a2 (1)**: 作业在加载模型权重时，自定义容器实现执行失败，导致进程终止。日志显示模型加载到0%时出现错误，可能是容器环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711838

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分省略，末尾仅显示plog备份、Node警告和清理，未出现测试执行或明确错误信息，需完整日志判断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279712036

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30668069293/job/91279711808) |


## [Run #30667859802](https://github.com/sgl-project/sglang/actions/runs/30667859802)
- **分支**: `main`
- **总耗时**: 8.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30667859802

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072652) |
| stage-b-test-16-npu-a3 | 6.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072659) |
| stage-b-test-1-npu-a2 (1) | 7.1min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072676) |
| stage-b-test-4-npu-a3 | 7.0min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072679) |
| stage-b-test-2-npu-a2 (1) | 7.0min | 环境问题 | NPU服务启动后健康检查持续返回503，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072684) |
| stage-b-test-2-npu-a2 (0) | 6.9min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072696) |
| stage-b-test-1-npu-a2 (0) | 7.0min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072698) |
| multimodal-gen-test-2-npu-a3 | 5.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072711) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.2min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279073019) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node 20弃用警告和上传artifact时无文件提示，无法判断失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072652

- **stage-b-test-16-npu-a3**: 日志显示在加载模型分片时，自定义容器实现执行失败，提示联系自托管 runner 管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072659

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行到TestAscendSamplingBackend.test_mmlu时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072676

- **stage-b-test-4-npu-a3**: 日志显示在运行DeepSeek-V2-Lite-W8A8模型测试时，Prefill阶段正常，但随后出现'Executing the custom container implementation failed'错误，提示联系runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072679

- **stage-b-test-2-npu-a2 (1)**: 服务在21:56:02启动成功，但/health_generate接口连续返回503 Service Unavailable，表明模型未就绪或推理异常，最终容器实现执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072684

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行正常，但中途出现'Executing the custom container implementation failed'错误，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072696

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到约1%进度时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072698

- **multimodal-gen-test-2-npu-a3**: 日志截断，仅显示checkout和upload-artifact操作，未包含multimodal-gen-test实际执行内容，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072711

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志截断，缺少测试运行阶段的关键输出，无法判断失败原因。可能为基础设施问题或日志采集异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279073019

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667859802/job/91279072702) |


## [Run #30667313690](https://github.com/sgl-project/sglang/actions/runs/30667313690)
- **分支**: `cctry-pd-health-retracted-queue`
- **总耗时**: 43.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30667313690

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 35.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435396) |

- **multimodal-gen-test-2-npu-a3**: 日志截断，缺少测试执行部分。仅见Node 20弃用警告、上传diffusion-failures目录时无文件，无法判断具体失败原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435396

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435344) |
| stage-b-test-1-npu-a2 (1) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435346) |
| stage-b-test-1-npu-a2 (0) | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435357) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435359) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435381) |
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435387) |
| stage-b-test-16-npu-a3 | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435401) |
| multimodal-gen-test-1-npu-a3 | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435425) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30667313690/job/91277435787) |


## [Run #30666364144](https://github.com/sgl-project/sglang/actions/runs/30666364144)
- **分支**: `main`
- **总耗时**: 25.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30666364144

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 24.0min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432338) |
| multimodal-gen-test-1-npu-a3 | 23.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432351) |
| stage-b-test-1-npu-a2 (0) | 24.2min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432384) |
| stage-b-test-1-npu-a2 (1) | 24.1min | 超时 | NPU测试作业因Scheduler watchdog超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432395) |
| multimodal-gen-test-2-npu-a3 | 24.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432419) |

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常（进度95%），但runner在21:48:35报错“Executing the custom container implementation failed”，提示联系管理员，属于runner环境或容器执行问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432338

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432351

- **stage-b-test-1-npu-a2 (0)**: 在运行test_npu_autoround_moe.py时，自定义容器实现执行失败，导致测试中断。日志显示为runner环境问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432384

- **stage-b-test-1-npu-a2 (1)**: 日志显示Scheduler watchdog timeout (self.watchdog_timeout=300)，调度器在300秒内无响应，导致作业被终止。可能因NPU资源竞争或负载过高引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432395

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions运行器初始化、依赖下载及上传artifact（无文件）等常规步骤，未包含multimodal测试执行或失败的具体输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432419

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274235985) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432347) |
| stage-b-test-16-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432363) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432389) |
| stage-b-test-2-npu-a2 (1) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432402) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30666364144/job/91274432790) |


---
*Auto-generated by npu_pr_monitor.py*