# NPU CI 执行监控
**生成时间**: 2026-08-08 00:16 UTC
**分析 Run 数**: 26

---

## [Run #30659015983](https://github.com/sgl-project/sglang/actions/runs/30659015983)
- **分支**: `feat/capture_graph`
- **总耗时**: 17.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30659015983

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 4.9min | 环境问题 | 自定义容器执行失败，模型加载过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556116) |
| stage-b-test-4-npu-a3 | 16.2min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556124) |
| stage-b-test-1-npu-a2 (0) | 16.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556140) |
| multimodal-gen-test-1-npu-a3 | 10.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556154) |
| stage-b-test-1-npu-a2 (1) | 16.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556175) |
| multimodal-gen-test-2-npu-a3 | 15.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556210) |
| stage-b-test-2-npu-a2 (1) | 16.1min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556246) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.5min | 其他 | 作业日志被截断，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556742) |

- **stage-b-test-16-npu-a3**: 作业在加载模型shards时（约14/161）报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556116

- **stage-b-test-4-npu-a3**: 日志显示测试运行到99%时，GitHub Actions报错“Executing the custom container implementation failed”，属于自托管runner环境问题，导致作业异常终止，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556124

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556140

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、checkout、上传artifact等步骤，未显示multimodal-gen测试的实际执行和失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556154

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常，但突然出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556175

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示Node.js 20弃用警告和上传artifact时无文件。可能因日志截断或作业在测试前被取消，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556210

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试进行到1316/1319时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556246

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分省略，无法看到测试命令输出或失败原因。仅显示上传metrics.json失败（文件不存在）及Node.js 20弃用警告，属于环境或日志收集问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556742

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556103) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30659015983/job/91250556161) |


## [Run #30657317260](https://github.com/sgl-project/sglang/actions/runs/30657317260)
- **分支**: `main`
- **总耗时**: 16.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30657317260

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 13.2min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031307) |
| stage-b-test-2-npu-a2 (0) | 15.0min | 环境问题 | 自定义容器执行失败，测试进程被中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031342) |
| multimodal-gen-test-2-npu-a3 | 12.6min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031360) |
| stage-b-test-1-npu-a2 (1) | 14.9min | 环境问题 | 自定义容器执行失败，NPU图捕获后作业异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031364) |
| multimodal-gen-test-1-npu-a3 | 12.0min | 其他 | 日志不完整，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031375) |
| stage-b-test-2-npu-a2 (1) | 15.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031376) |
| stage-b-test-4-npu-a3 | 12.4min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031387) |
| stage-b-test-1-npu-a2 (0) | 14.7min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031418) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.6min | 其他 | 日志被截断，无法看到实际测试结果，仅显示作业正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031638) |

- **stage-b-test-16-npu-a3**: 作业在启动DeepSeek-R1模型服务时，自定义容器实现执行失败，可能是NPU资源分配或容器配置问题，导致服务无法正常启动。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031307

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行到1314/1319时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031342

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031360

- **stage-b-test-1-npu-a2 (1)**: 日志显示NPU图捕获正常完成，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031364

- **multimodal-gen-test-1-npu-a3**: 作业日志被截断，仅显示上传diffusion-failures目录时无文件，未包含测试执行和失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031375

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试进行到1307/1319时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031376

- **stage-b-test-4-npu-a3**: 作业在加载模型分片（50%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031387

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试在运行约1分钟后，执行自定义容器实现时失败（Executing the custom container implementation failed），可能是NPU环境或容器问题导致作业中断，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031418

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分被省略，未包含测试执行的关键输出（如性能指标、错误信息等），仅显示上传metrics.json失败（文件不存在）及后续清理步骤，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031638

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657317260/job/91245031353) |


## [Run #30657257972](https://github.com/sgl-project/sglang/actions/runs/30657257972)
- **分支**: `fix/video-decoder-pin-memory-fallback`
- **总耗时**: 43.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30657257972

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 32.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244711957) |

- **multimodal-gen-test-2-npu-a3**: 日志中仅显示Node 20弃用警告、上传diffusion-failures目录无文件等提示，未包含测试执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244711957

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244711907) |
| stage-b-test-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244711918) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244711934) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244711963) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244711966) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244711974) |
| multimodal-gen-test-1-npu-a3 | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244712000) |
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244712008) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30657257972/job/91244712448) |


## [Run #30656760737](https://github.com/sgl-project/sglang/actions/runs/30656760737)
- **分支**: `qiaolin_replayssm_cutedsl_verify`
- **总耗时**: 43.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30656760737

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 27.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243275986) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.5min | 环境问题 | 作业日志被截断，未显示实际失败原因，仅看到正常清理流程。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243276312) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业整体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243275986

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分被省略，无法定位具体错误。作业在运行后进入清理阶段，可能因测试失败或环境异常提前结束，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243276312

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243275881) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243275893) |
| multimodal-gen-test-1-npu-a3 | 34.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243275902) |
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243275954) |
| stage-b-test-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243275957) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243275973) |
| stage-b-test-1-npu-a2 (0) | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243276000) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656760737/job/91243276032) |


## [Run #30656523643](https://github.com/sgl-project/sglang/actions/runs/30656523643)
- **分支**: `main`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30656523643

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 4.3min | 环境问题 | 自定义容器执行失败，测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308855) |
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308876) |
| multimodal-gen-test-2-npu-a3 | 4.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308880) |
| stage-b-test-2-npu-a2 (1) | 4.2min | 环境问题 | 自定义容器执行失败，测试进程启动后立即崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308891) |
| stage-b-test-16-npu-a3 | 3.4min | 环境问题 | 自定义容器执行失败，测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308942) |
| stage-b-test-2-npu-a2 (0) | 4.1min | 环境问题 | 自定义容器执行失败，导致测试未运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308949) |
| stage-b-test-1-npu-a2 (1) | 4.1min | 环境问题 | 自定义容器执行失败，测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308957) |
| stage-b-test-1-npu-a2 (0) | 4.3min | 环境问题 | 自定义容器执行失败，导致测试未运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308962) |
| stage-b-test-4-npu-a3 | 4.4min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308975) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.4min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242309539) |

- **stage-a-unit-test-npu**: 作业在启动NPU测试时，自定义容器实现执行失败（Executing the custom container implementation failed），导致测试进程未完成即终止，属于自托管runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308855

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308876

- **multimodal-gen-test-2-npu-a3**: 日志被截断，中间省略了关键测试步骤。可见部分仅显示runner启动、checkout、upload-artifact（无文件上传）及Node 20弃用警告，未出现测试命令或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308880

- **stage-b-test-2-npu-a2 (1)**: 测试test_npu_mla_fia_w8a8int8.py启动后，在运行test_a_gsm8k时，自定义容器实现执行失败，可能是NPU环境或容器配置问题，导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308891

- **stage-b-test-16-npu-a3**: 作业在启动测试命令后立即报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308942

- **stage-b-test-2-npu-a2 (0)**: 日志显示在运行测试命令后，出现错误'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308949

- **stage-b-test-1-npu-a2 (1)**: 作业在启动第一个测试时，自定义容器实现执行失败（Executing the custom container implementation failed），导致测试进程未能启动，属于NPU自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308957

- **stage-b-test-1-npu-a2 (0)**: 日志显示在运行测试命令后，自定义容器实现执行失败（Executing the custom container implementation failed），可能是K8s容器环境或NPU资源分配问题，测试未实际执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308962

- **stage-b-test-4-npu-a3**: 作业在加载模型权重时（Multi-thread loading shards 0%）自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242308975

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业初始化、环境准备和清理步骤，未展示测试运行过程及错误信息，可能因日志截断或测试未实际执行导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656523643/job/91242309539


## [Run #30656086133](https://github.com/sgl-project/sglang/actions/runs/30656086133)
- **分支**: `fix/staging-radix-grid-align`
- **总耗时**: 43.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30656086133

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 27.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935450) |
| multimodal-gen-test-1-npu-a3 | 34.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935453) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935450

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾仅显示上传diffusion-failures目录时未找到文件，说明测试可能未生成失败产物，但根本原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935453

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240705109) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935361) |
| stage-b-test-1-npu-a2 (0) | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935394) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935401) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935429) |
| stage-b-test-16-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935482) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935490) |
| stage-b-test-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935500) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30656086133/job/91240935998) |


## [Run #30655704195](https://github.com/sgl-project/sglang/actions/runs/30655704195)
- **分支**: `cp/interleave-v2-fix`
- **总耗时**: 41.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30655704195

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 33.7min | 代码错误 | NPU量化测试test_npu_autoround_moe.py失败，导致作业整体失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626375) |
| multimodal-gen-test-2-npu-a3 | 25.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626380) |

- **stage-b-test-1-npu-a2 (0)**: 测试套件中3/5通过，但quant/test_npu_autoround_moe.py返回退出码1，耗时620秒。该测试涉及AutoRound MoE量化功能，可能因代码逻辑或环境配置问题导致失败，需检查具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626375

- **multimodal-gen-test-2-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未生成失败样本，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626380

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626346) |
| multimodal-gen-test-1-npu-a3 | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626351) |
| stage-b-test-1-npu-a2 (1) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626355) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626356) |
| stage-b-test-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626370) |
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626406) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626424) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30655704195/job/91239626931) |


## [Run #30652429019](https://github.com/sgl-project/sglang/actions/runs/30652429019)
- **分支**: `jialino/trtllm-mha-war-slot-snapshot`
- **总耗时**: 45.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30652429019

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 23.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809302) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。结尾显示上传diffusion-failures目录时未找到文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809302

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809261) |
| stage-b-test-1-npu-a2 (0) | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809264) |
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809276) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809303) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809305) |
| multimodal-gen-test-1-npu-a3 | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809306) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809323) |
| stage-b-test-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809349) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30652429019/job/91228809534) |


## [Run #30650805556](https://github.com/sgl-project/sglang/actions/runs/30650805556)
- **分支**: `fp8-hybrid-m32-dispatch`
- **总耗时**: 42.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30650805556

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172313) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或日志被截断，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172313

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172234) |
| multimodal-gen-test-1-npu-a3 | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172240) |
| stage-b-test-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172247) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172257) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172277) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172289) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172303) |
| stage-b-test-1-npu-a2 (0) | 41.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249172336) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30650805556/job/91249173509) |


## [Run #30649896484](https://github.com/sgl-project/sglang/actions/runs/30649896484)
- **分支**: `feat/windowed-mtp-draft-decode`
- **总耗时**: 65.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30649896484

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376726) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误或失败断言，仅显示Node.js 20弃用警告和上传工件时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376726

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376696) |
| stage-b-test-16-npu-a3 | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376705) |
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376729) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376745) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376754) |
| stage-b-test-1-npu-a2 (1) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376769) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376770) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220376793) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30649896484/job/91220377154) |


## [Run #30646214427](https://github.com/sgl-project/sglang/actions/runs/30646214427)
- **分支**: `codex/diffusion-kv-gather-sp`
- **总耗时**: 30.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30646214427

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 28.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30646214427/job/91208416142) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有checkout、upload-artifact等步骤，且upload-artifact提示未找到diffusion-failures目录，说明测试可能未运行或提前结束，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30646214427/job/91208416142

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30646214427/job/91208233582) |
| multimodal-gen-test-1-npu-a3 | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30646214427/job/91208416131) |


## [Run #30644204797](https://github.com/sgl-project/sglang/actions/runs/30644204797)
- **分支**: `fix/video-decoder-pin-memory-fallback`
- **总耗时**: 62.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30644204797

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 27.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803121) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅显示Node.js 20弃用警告、上传artifact时无文件等非关键信息。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803121

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202802945) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803024) |
| multimodal-gen-test-1-npu-a3 | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803031) |
| stage-b-test-1-npu-a2 (0) | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803046) |
| stage-b-test-4-npu-a3 | 39.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803067) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803082) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803088) |
| stage-b-test-1-npu-a2 (1) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803208) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30644204797/job/91202803795) |


## [Run #30641897415](https://github.com/sgl-project/sglang/actions/runs/30641897415)
- **分支**: `main`
- **总耗时**: 56.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30641897415

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749570) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node 20弃用警告和上传artifact时无文件。可能因日志截断或测试未实际运行，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749570

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 43.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749547) |
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749561) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749577) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749626) |
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749658) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749684) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749709) |
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749731) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30641897415/job/91193749840) |


## [Run #30640913599](https://github.com/sgl-project/sglang/actions/runs/30640913599)
- **分支**: `sam/prompt-logprob-fast-topk`
- **总耗时**: 66.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30640913599

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 46.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449780) |
| multimodal-gen-test-2-npu-a3 | 27.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449964) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node.js 20弃用警告和上传diffusion-failures产物时未找到文件的提示，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449780

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449964

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449695) |
| stage-b-test-1-npu-a2 (0) | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449719) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449738) |
| stage-b-test-16-npu-a3 | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449784) |
| stage-b-test-1-npu-a2 (1) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449795) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449824) |
| stage-b-test-2-npu-a2 (1) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190449890) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640913599/job/91190450386) |


## [Run #30640635366](https://github.com/sgl-project/sglang/actions/runs/30640635366)
- **分支**: `sam/flat-raw-topk-sched`
- **总耗时**: 47.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30640635366

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 27.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505704) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未出现测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505704

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505620) |
| stage-b-test-4-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505631) |
| stage-a-unit-test-npu | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505646) |
| stage-b-test-1-npu-a2 (0) | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505669) |
| stage-b-test-16-npu-a3 | 19.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505673) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505718) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505772) |
| stage-b-test-1-npu-a2 (1) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189505776) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640635366/job/91189506142) |


## [Run #30640020256](https://github.com/sgl-project/sglang/actions/runs/30640020256)
- **分支**: `xinyuan/fix-verify-mask-fixture`
- **总耗时**: 49.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30640020256

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 27.7min | 其他 | 作业正常结束，无失败迹象，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356538) |

- **multimodal-gen-test-2-npu-a3**: 日志显示作业流程完整执行，最后上传diffusion-failures目录时提示无文件，属正常情况（无失败用例）。未发现测试失败、超时或环境错误，作业可能因其他原因被标记为失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356538

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356530) |
| stage-b-test-1-npu-a2 (1) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356534) |
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356537) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356580) |
| stage-b-test-1-npu-a2 (0) | 44.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356588) |
| stage-b-test-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356601) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356606) |
| multimodal-gen-test-1-npu-a3 | 46.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356611) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30640020256/job/91188356939) |


## [Run #30635826053](https://github.com/sgl-project/sglang/actions/runs/30635826053)
- **分支**: `main`
- **总耗时**: 83.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30635826053

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202231) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202231

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202130) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202153) |
| stage-b-test-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202160) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202177) |
| stage-b-test-16-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202182) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202185) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202194) |
| stage-b-test-1-npu-a2 (1) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202249) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635826053/job/91173202620) |


## [Run #30635135108](https://github.com/sgl-project/sglang/actions/runs/30635135108)
- **分支**: `index-cache-poc`
- **总耗时**: 94.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30635135108

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 29.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传工件信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856647) |

- **multimodal-gen-test-2-npu-a3**: 日志中仅包含Node.js版本弃用警告、上传工件时未找到diffusion-failures目录等提示，未出现测试失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856647

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 44.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856297) |
| stage-b-test-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856299) |
| stage-b-test-16-npu-a3 | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856302) |
| multimodal-gen-test-1-npu-a3 | 33.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856334) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856336) |
| stage-b-test-2-npu-a2 (0) | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856394) |
| stage-b-test-1-npu-a2 (1) | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856426) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170856492) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30635135108/job/91170857214) |


## [Run #30634852096](https://github.com/sgl-project/sglang/actions/runs/30634852096)
- **分支**: `mtp-draft-sidecar-pools`
- **总耗时**: 79.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30634852096

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 6.7min | 代码错误 | HiCache MLA测试失败，服务启动后测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834174) |
| stage-b-test-1-npu-a2 (0) | 6.4min | 代码错误 | HiCache MHA测试失败，服务启动后测试用例返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834199) |
| multimodal-gen-test-2-npu-a3 | 25.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834277) |
| multimodal-gen-test-1-npu-a3 | 42.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834438) |

- **stage-b-test-4-npu-a3**: test_npu_hicache_mla.py测试失败，0/5通过。服务启动命令正常，但测试执行时返回错误，可能是HiCache功能或MLA相关代码存在缺陷，需检查测试日志定位具体断言失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834174

- **stage-b-test-1-npu-a2 (0)**: 测试test_npu_hicache_mha.py在启动Qwen2.5-7B-Instruct服务后执行失败，0/5测试通过，可能因HiCache功能实现或配置问题导致，需检查相关代码。
  链接: https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834199

- **multimodal-gen-test-2-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未包含多模态生成测试的具体执行结果或错误输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834277

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示GitHub Actions环境准备、Node.js弃用警告及上传artifact时无文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834438

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834219) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834237) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834259) |
| stage-b-test-2-npu-a2 (0) | 16.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834331) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834442) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634852096/job/91169834849) |


## [Run #30634162166](https://github.com/sgl-project/sglang/actions/runs/30634162166)
- **分支**: `fuse-swiglu-moe-up-gemm-epilogue`
- **总耗时**: 80.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30634162166

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.7min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525504) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js 20弃用警告和上传失败提示（diffusion-failures目录无文件），无法判断具体失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525504

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525496) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525500) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525522) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525528) |
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525531) |
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525536) |
| multimodal-gen-test-1-npu-a3 | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525540) |
| stage-b-test-1-npu-a2 (1) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525667) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30634162166/job/91167525964) |


## [Run #30633590654](https://github.com/sgl-project/sglang/actions/runs/30633590654)
- **分支**: `agent/sana-video-t2v`
- **总耗时**: 56.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30633590654

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 32.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633590654/job/91165668797) |

- **multimodal-gen-test-2-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未生成失败样本，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633590654/job/91165668797

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633590654/job/91165668758) |


## [Run #30633567694](https://github.com/sgl-project/sglang/actions/runs/30633567694)
- **分支**: `xinyuan/wide-row-silu-clamp`
- **总耗时**: 73.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30633567694

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633567694/job/91165592155) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633567694/job/91165592213) |
| stage-b-test-1-npu-a2 (1) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633567694/job/91165592224) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633567694/job/91165592236) |
| stage-b-test-4-npu-a3 | 38.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633567694/job/91165592257) |
| stage-b-test-16-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633567694/job/91165592309) |
| stage-b-test-2-npu-a2 (1) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633567694/job/91165592405) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633567694/job/91165592795) |


## [Run #30633530169](https://github.com/sgl-project/sglang/actions/runs/30633530169)
- **分支**: `bbuf/kimi-k3-standalone-kernels`
- **总耗时**: 48.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30633530169

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 3.3min | 代码错误 | 测试文件缺少主入口导致收集失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506001) |
| stage-b-test-2-npu-a2 (1) | 4.2min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506010) |
| stage-b-test-4-npu-a3 | 3.3min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506039) |
| stage-b-test-1-npu-a2 (1) | 4.3min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506073) |
| stage-b-test-1-npu-a2 (0) | 4.2min | 代码错误 | 测试文件缺少main入口导致CI收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506076) |
| multimodal-gen-test-2-npu-a3 | 31.7min | 环境问题 | 作业因缺少失败产物文件而提前结束，未显示实际测试失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506112) |
| stage-b-test-2-npu-a2 (0) | 3.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506191) |
| stage-a-unit-test-npu | 4.0min | 代码错误 | 测试文件缺少入口导致收集失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506209) |

- **stage-b-test-16-npu-a3**: test/registered/kernels/ops/attention/test_kda_fused_decode.py 缺少 `if __name__ == "__main__":` 入口，pytest 风格测试在直接运行时会被静默跳过，CI 收集测试时检测到该问题并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506001

- **stage-b-test-2-npu-a2 (1)**: test_kda_fused_decode.py缺少`if __name__ == "__main__":`入口，pytest风格测试在`python3 file.py -f`下会静默跳过，CI的collect_tests检查抛出ValueError，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506010

- **stage-b-test-4-npu-a3**: test_kda_fused_decode.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在直接运行时被静默跳过，CI 收集测试时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506039

- **stage-b-test-1-npu-a2 (1)**: test_kda_fused_decode.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 的 collect_tests 检查抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506073

- **stage-b-test-1-npu-a2 (0)**: test_kda_fused_decode.py未添加`if __name__ == "__main__":`入口，pytest风格测试在直接运行时会被静默跳过，CI的collect_tests检查强制要求该入口，导致脚本退出码非零。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506076

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures目录时提示无文件，作业正常结束但无测试结果，可能因测试未运行或产物路径错误，需检查测试执行阶段日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506112

- **stage-b-test-2-npu-a2 (0)**: 作业在运行测试命令时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506191

- **stage-a-unit-test-npu**: test_kda_fused_decode.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 收集测试时抛出 ValueError 并退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506209

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165505999) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30633530169/job/91165506312) |


## [Run #30632399539](https://github.com/sgl-project/sglang/actions/runs/30632399539)
- **分支**: `qwen3.5_integration_gfx950_fmha_fp8_hd256`
- **总耗时**: 82.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30632399539

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764346) |
| multimodal-gen-test-1-npu-a3 | 35.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764379) |

- **multimodal-gen-test-2-npu-a3**: 日志被截断，中间省略了关键测试步骤。可见部分仅显示runner启动、checkout、上传artifact（无文件）及清理流程，无测试命令或错误输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764346

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764379

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764324) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764398) |
| stage-b-test-1-npu-a2 (1) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764424) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764433) |
| stage-b-test-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764477) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764482) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764770) |
| stage-b-test-2-npu-a2 (1) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30632399539/job/91161764858) |


## [Run #30631665545](https://github.com/sgl-project/sglang/actions/runs/30631665545)
- **分支**: `main`
- **总耗时**: 63.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30631665545

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 12.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739624) |
| stage-b-test-1-npu-a2 (0) | 25.5min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739627) |
| multimodal-gen-test-2-npu-a3 | 26.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739714) |

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常（进度82%），但突然报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739624

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重时，自定义容器实现执行失败，导致进程终止。可能是NPU环境或容器配置问题，而非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739627

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业整体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739714

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739556) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739598) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739602) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739628) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739694) |
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159739727) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30631665545/job/91159740043) |


## [Run #30630833437](https://github.com/sgl-project/sglang/actions/runs/30630833437)
- **分支**: `bbuf/kimi-k3-standalone-kernels`
- **总耗时**: 42.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30630833437

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628907) |
| stage-a-unit-test-npu | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628908) |
| stage-b-test-1-npu-a2 (1) | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628920) |
| stage-b-test-1-npu-a2 (0) | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628945) |
| stage-b-test-4-npu-a3 | 2.9min | 代码错误 | 测试文件缺少main入口导致CI收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628947) |
| multimodal-gen-test-1-npu-a3 | 27.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628971) |
| stage-b-test-2-npu-a2 (0) | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628977) |
| stage-b-test-16-npu-a3 | 4.1min | 代码错误 | 测试文件缺少主入口导致收集失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628991) |
| stage-b-test-2-npu-a2 (1) | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156629004) |

- **multimodal-gen-test-2-npu-a3**: 日志被截断，中间省略了关键测试输出。可见部分仅显示Node 20弃用警告、上传diffusion-failures目录时无文件，以及清理步骤，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628907

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628908

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628920

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628945

- **stage-b-test-4-npu-a3**: test/registered/kernels/ops/attention/test_kda_fused_decode.py缺少`if __name__ == "__main__":`入口，导致pytest风格测试在直接运行时静默跳过，CI的collect_tests检查抛出ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628947

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但无法确认具体失败点，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628971

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是存储配置错误或上游任务未成功生成该 blob。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628977

- **stage-b-test-16-npu-a3**: test_kda_fused_decode.py 缺少 `if __name__ == "__main__":` 入口，pytest 风格测试在直接运行时会被静默跳过，CI 收集测试时检测到该问题并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156628991

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156629004

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30630833437/job/91156629249) |


---
*Auto-generated by npu_pr_monitor.py*