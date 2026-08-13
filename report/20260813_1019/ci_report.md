# NPU CI 执行监控
**生成时间**: 2026-08-13 02:19 UTC
**分析 Run 数**: 32

---

## [Run #30494421911](https://github.com/sgl-project/sglang/actions/runs/30494421911)
- **分支**: `main`
- **总耗时**: 5.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30494421911

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 4.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905325) |
| multimodal-gen-test-1-npu-a3 | 4.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905352) |
| multimodal-gen-test-2-npu-a3 | 4.8min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905356) |
| stage-b-test-4-npu-a3 | 3.6min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905358) |
| stage-b-test-1-npu-a2 (1) | 4.8min | 环境问题 | 自定义容器执行失败，测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905371) |
| stage-b-test-1-npu-a2 (0) | 4.8min | 环境问题 | 自定义容器执行失败，测试启动后立即崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905372) |
| stage-b-test-2-npu-a2 (0) | 4.8min | 环境问题 | 自定义容器执行失败，测试未完成即中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905373) |
| stage-b-test-2-npu-a2 (1) | 4.8min | 环境问题 | 自定义容器执行失败，测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905407) |
| stage-b-test-16-npu-a3 | 4.8min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905410) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 1.4min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到作业启动和清理过程。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905642) |

- **stage-a-unit-test-npu**: 测试在运行第二个NPU测试文件时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905325

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js 20弃用警告和上传artifact时无文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905352

- **multimodal-gen-test-2-npu-a3**: 作业在运行multimodal-gen-test后上传diffusion-failures目录时提示无文件，说明测试可能通过或失败未生成产物，但关键测试日志被省略，无法定位具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905356

- **stage-b-test-4-npu-a3**: 日志显示模型加载时出现pad_token_id和generation_config警告，随后自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905358

- **stage-b-test-1-npu-a2 (1)**: 作业在启动第一个测试后立即报错，错误为自定义容器实现执行失败，属于runner环境或容器配置问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905371

- **stage-b-test-1-npu-a2 (0)**: 作业在运行第一个测试test_npu_hicache_mha.py时，自定义容器实现执行失败，错误提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905372

- **stage-b-test-2-npu-a2 (0)**: 作业在运行test_npu_graph_tp2_bf16.py时，自定义容器实现执行失败（Executing the custom container implementation failed），导致测试进程被终止，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905373

- **stage-b-test-2-npu-a2 (1)**: 作业在启动测试后立即报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905407

- **stage-b-test-16-npu-a3**: 日志显示TP/EP各进程开始加载权重后，出现"Executing the custom container implementation failed"错误，可能是容器环境或NPU资源问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905410

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分被省略，无法定位具体失败点。从可见内容看，作业正常启动、执行了plog备份和清理，未发现明确错误信息，可能因日志不完整导致无法判断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30494421911/job/90719905642


## [Run #30493992786](https://github.com/sgl-project/sglang/actions/runs/30493992786)
- **分支**: `qiaolin_replayssm`
- **总耗时**: 10.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30493992786

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 10.3min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509410) |
| stage-b-test-2-npu-a2 (1) | 10.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509413) |
| multimodal-gen-test-2-npu-a3 | 10.3min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509415) |
| multimodal-gen-test-1-npu-a3 | 10.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509418) |
| stage-b-test-16-npu-a3 | 10.3min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509421) |
| stage-b-test-1-npu-a2 (0) | 10.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509422) |
| stage-b-test-1-npu-a2 (1) | 10.3min | 环境问题 | 自定义容器执行失败，NPU测试在运行中异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509477) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.4min | 其他 | 作业日志不完整，未显示实际失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509608) |
| stage-b-test-4-npu-a3 | 10.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718510152) |

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行至79%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509410

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常，但在22:02:39时出现错误'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509413

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含实际测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509415

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未展示multimodal-gen测试的具体执行和失败输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509418

- **stage-b-test-16-npu-a3**: 日志显示模型分片加载到75%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509421

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常（进度93%），但突然出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509422

- **stage-b-test-1-npu-a2 (1)**: 日志显示sglang服务正常启动并处理请求，但在测试进行到1/1319时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509477

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中未包含测试执行阶段的错误信息，仅显示runner启动、依赖下载、plog备份和清理步骤。可能因日志截断或测试未实际运行而失败，需查看完整日志以确定具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509608

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常，但最后出现“Executing the custom container implementation failed”错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718510152

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30493992786/job/90718509462) |


## [Run #30491166702](https://github.com/sgl-project/sglang/actions/runs/30491166702)
- **分支**: `vincent/lfm2-tool-parser-json-literals-minimal`
- **总耗时**: 43.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30491166702

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198901) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198910) |
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198914) |
| stage-b-test-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198915) |
| multimodal-gen-test-1-npu-a3 | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198920) |
| multimodal-gen-test-2-npu-a3 | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198933) |
| stage-b-test-1-npu-a2 (0) | 43.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198934) |
| stage-b-test-1-npu-a2 (1) | 31.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198944) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709198960) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30491166702/job/90709199489) |


## [Run #30490798527](https://github.com/sgl-project/sglang/actions/runs/30490798527)
- **分支**: `main`
- **总耗时**: 42.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30490798527

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997756) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997767) |
| multimodal-gen-test-2-npu-a3 | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997769) |
| multimodal-gen-test-1-npu-a3 | 26.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997773) |
| stage-b-test-16-npu-a3 | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997774) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997795) |
| stage-b-test-4-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997797) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997809) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707997814) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30490798527/job/90707998141) |


## [Run #30488678642](https://github.com/sgl-project/sglang/actions/runs/30488678642)
- **分支**: `fix-per-layer-head-count-attention-backends`
- **总耗时**: 42.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30488678642

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238668) |
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238673) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238677) |
| stage-b-test-4-npu-a3 | 41.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238705) |
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238716) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238746) |
| multimodal-gen-test-2-npu-a3 | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238755) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238769) |
| stage-b-test-1-npu-a2 (0) | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704238796) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30488678642/job/90704239232) |


## [Run #30484003513](https://github.com/sgl-project/sglang/actions/runs/30484003513)
- **分支**: `jthomson04/kv-event-coalesce`
- **总耗时**: 50.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30484003513

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 42.7min | 其他 | 作业日志被截断，未显示实际失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145266) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，但未出现明确错误或测试失败信息，可能为作业被提前终止或日志不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145266

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 41.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145262) |
| multimodal-gen-test-1-npu-a3 | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145300) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145330) |
| stage-b-test-1-npu-a2 (1) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145331) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145339) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145353) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145372) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145468) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30484003513/job/90685145968) |


## [Run #30483636541](https://github.com/sgl-project/sglang/actions/runs/30483636541)
- **分支**: `main`
- **总耗时**: 51.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30483636541

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 43.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980761) |
| multimodal-gen-test-1-npu-a3 | 49.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980795) |
| multimodal-gen-test-2-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980796) |
| stage-b-test-4-npu-a3 | 40.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980805) |
| stage-b-test-16-npu-a3 | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980814) |
| stage-b-test-1-npu-a2 (1) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980830) |
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980843) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980864) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683980876) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483636541/job/90683981437) |


## [Run #30483321366](https://github.com/sgl-project/sglang/actions/runs/30483321366)
- **分支**: `swa-admission-never-fits`
- **总耗时**: 53.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30483321366

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803720) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803765) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803805) |
| stage-b-test-4-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803807) |
| stage-b-test-2-npu-a2 (0) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803809) |
| multimodal-gen-test-1-npu-a3 | 33.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803811) |
| stage-b-test-1-npu-a2 (0) | 43.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803825) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803846) |
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682803861) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30483321366/job/90682804287) |


## [Run #30482542095](https://github.com/sgl-project/sglang/actions/runs/30482542095)
- **分支**: `main`
- **总耗时**: 15.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30482542095

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 14.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146377) |
| stage-b-test-1-npu-a2 (1) | 14.4min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重后崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146413) |
| stage-b-test-16-npu-a3 | 14.4min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146422) |
| multimodal-gen-test-1-npu-a3 | 6.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146446) |
| stage-b-test-4-npu-a3 | 14.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146462) |
| stage-b-test-1-npu-a2 (0) | 14.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146466) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.7min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146815) |
| multimodal-gen-test-2-npu-a3 | 14.3min | 其他 | 作业未显示明确失败原因，日志仅包含正常执行和警告信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680193925) |
| stage-b-test-2-npu-a2 (1) | 14.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680270807) |

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行正常（进度53%），但突然报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146377

- **stage-b-test-1-npu-a2 (1)**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员，可能因NPU环境或容器配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146413

- **stage-b-test-16-npu-a3**: 日志显示测试运行到84%时，runner报错“Executing the custom container implementation failed”，随后作业终止。这属于runner环境或容器配置问题，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146422

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node.js弃用警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146446

- **stage-b-test-4-npu-a3**: 日志显示测试运行中（Prefill batch正常），但突然报错"Executing the custom container implementation failed"，提示联系runner管理员，属于runner容器环境故障，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146462

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试在捕获批次完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146466

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业初始化和清理步骤，缺少测试运行及失败关键信息，可能因日志截断或作业提前终止导致，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146815

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败或错误信息，仅有Node 20弃用警告和上传artifact时未找到文件的提示，作业可能因其他未记录原因失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680193925

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行到98%时，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680270807

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30482542095/job/90680146379) |


## [Run #30481631146](https://github.com/sgl-project/sglang/actions/runs/30481631146)
- **分支**: `optimize-mem-pool-slot-alloc`
- **总耗时**: 43.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30481631146

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122264) |
| stage-b-test-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122270) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122318) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122365) |
| stage-b-test-1-npu-a2 (1) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122368) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122371) |
| stage-b-test-2-npu-a2 (0) | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122372) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122415) |
| multimodal-gen-test-2-npu-a3 | 34.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677122421) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30481631146/job/90677123215) |


## [Run #30479687961](https://github.com/sgl-project/sglang/actions/runs/30479687961)
- **分支**: `jthomson04/sglang-cache-salt-events`
- **总耗时**: 58.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30479687961

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 2.6min | 环境问题 | pip安装依赖时网络连接中断，导致下载不完整。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305891) |
| stage-b-test-2-npu-a2 (1) | 6.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305965) |

- **stage-b-test-2-npu-a2 (0)**: 在安装Python包时，pip从远程下载文件过程中连接断开（IncompleteRead），仅读取25MB但预期需188MB，导致安装失败。属于网络不稳定或源服务器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305891

- **stage-b-test-2-npu-a2 (1)**: 作业在加载模型分片后，执行自定义容器实现时失败（错误：Executing the custom container implementation failed），可能因NPU驱动或CANN环境配置问题导致，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305965

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305880) |
| stage-b-test-4-npu-a3 | 40.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305886) |
| stage-a-unit-test-npu | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305888) |
| stage-b-test-1-npu-a2 (0) | 42.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305897) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670305933) |
| multimodal-gen-test-2-npu-a3 | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670306069) |
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670306221) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479687961/job/90670306622) |


## [Run #30479063985](https://github.com/sgl-project/sglang/actions/runs/30479063985)
- **分支**: `optimize-mem-pool-slot-alloc`
- **总耗时**: 34.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30479063985

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (1) | 14.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126146) |
| stage-b-test-4-npu-a3 | 33.6min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126156) |
| stage-b-test-1-npu-a2 (1) | 10.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126312) |
| stage-b-test-1-npu-a2 (0) | 14.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126391) |

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行到97%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126146

- **stage-b-test-4-npu-a3**: 日志显示测试在运行到83%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题，而非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126156

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126312

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126391

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126145) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126202) |
| multimodal-gen-test-1-npu-a3 | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126213) |
| stage-a-unit-test-npu | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126249) |
| multimodal-gen-test-2-npu-a3 | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126286) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479063985/job/90668126876) |


## [Run #30479008530](https://github.com/sgl-project/sglang/actions/runs/30479008530)
- **分支**: `pp-abort-followup`
- **总耗时**: 58.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30479008530

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 44.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062207) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到diffusion-failures目录。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062207

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062102) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062111) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062127) |
| multimodal-gen-test-1-npu-a3 | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062131) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062146) |
| stage-b-test-16-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062344) |
| stage-b-test-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062356) |
| stage-b-test-1-npu-a2 (1) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062556) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30479008530/job/90668062897) |


## [Run #30476169876](https://github.com/sgl-project/sglang/actions/runs/30476169876)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 46.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30476169876

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443648) |
| multimodal-gen-test-2-npu-a3 | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443719) |
| stage-b-test-2-npu-a2 (1) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443724) |
| stage-b-test-4-npu-a3 | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443742) |
| stage-b-test-16-npu-a3 | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443744) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443747) |
| multimodal-gen-test-1-npu-a3 | 37.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443759) |
| stage-b-test-1-npu-a2 (0) | 43.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443779) |
| stage-b-test-2-npu-a2 (0) | 16.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658443827) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476169876/job/90658444405) |


## [Run #30476084643](https://github.com/sgl-project/sglang/actions/runs/30476084643)
- **分支**: `feat/capture_graph`
- **总耗时**: 43.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30476084643

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 29.9min | 其他 | 日志未显示测试失败原因，仅显示上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156910) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行后上传diffusion-failures目录时提示无文件，未出现明确错误或超时信息，可能测试未执行或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156910

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156863) |
| multimodal-gen-test-2-npu-a3 | 32.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156874) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156888) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156925) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156943) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156965) |
| stage-b-test-16-npu-a3 | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156970) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658156971) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476084643/job/90658157528) |


## [Run #30476010772](https://github.com/sgl-project/sglang/actions/runs/30476010772)
- **分支**: `pp-abort-followup`
- **总耗时**: 17.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30476010772

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 16.6min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584288) |
| multimodal-gen-test-2-npu-a3 | 16.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584289) |
| multimodal-gen-test-1-npu-a3 | 16.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584290) |
| stage-b-test-16-npu-a3 | 16.6min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584315) |
| stage-b-test-1-npu-a2 (0) | 16.6min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584327) |
| stage-b-test-2-npu-a2 (1) | 16.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584339) |
| stage-b-test-4-npu-a3 | 16.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584349) |
| stage-b-test-2-npu-a2 (0) | 16.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584353) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | 其他 | 日志显示测试状态为pass，但作业最终失败，可能因后续清理或资源回收问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584688) |

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行到36%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU资源或容器环境不稳定导致，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584288

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584289

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有GitHub Actions环境准备、Node版本弃用警告及上传artifact时无文件提示，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584290

- **stage-b-test-16-npu-a3**: 日志显示在测试运行过程中，自定义容器实现执行失败（Executing the custom container implementation failed），提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584315

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到49%时，自定义容器实现执行失败（Executing the custom container implementation failed），导致作业中断。这属于自托管runner环境或容器问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584327

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584339

- **stage-b-test-4-npu-a3**: 日志显示测试运行中（进度46%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584349

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584353

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 测试本身通过（test_status: pass），失败发生在作业后处理阶段，涉及plog备份和k8s清理，可能是环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584688

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30476010772/job/90663584317) |


## [Run #30475234654](https://github.com/sgl-project/sglang/actions/runs/30475234654)
- **分支**: `gdn-qwen35-split-view`
- **总耗时**: 54.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30475234654

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655288854) |
| multimodal-gen-test-2-npu-a3 | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655288869) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655288889) |
| stage-b-test-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655288893) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655288897) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655288917) |
| stage-b-test-1-npu-a2 (1) | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655288998) |
| multimodal-gen-test-1-npu-a3 | 27.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655289016) |
| stage-b-test-2-npu-a2 (0) | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655289053) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30475234654/job/90655289818) |


## [Run #30474787504](https://github.com/sgl-project/sglang/actions/runs/30474787504)
- **分支**: `main`
- **总耗时**: 36.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30474787504

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 34.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908234) |
| stage-b-test-4-npu-a3 | 19.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化或运行异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908373) |

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到51%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908234

- **stage-b-test-4-npu-a3**: 作业在加载Qwen3模型权重后，自定义容器实现执行失败（Executing the custom container implementation failed），可能是NPU设备、Ascend环境或容器配置问题，导致测试无法继续。
  链接: https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908373

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908277) |
| stage-b-test-1-npu-a2 (1) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908304) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908311) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908344) |
| multimodal-gen-test-2-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908357) |
| stage-b-test-16-npu-a3 | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908363) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653908440) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30474787504/job/90653909018) |


## [Run #30470650322](https://github.com/sgl-project/sglang/actions/runs/30470650322)
- **分支**: `amd/dsv4-aiter-fused-mhc-cross-layer`
- **总耗时**: 81.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30470650322

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 40.1min | 其他 | 作业日志被截断，未显示实际失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842358) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅看到上传diffusion-failures目录时提示无文件，可能测试未生成失败产物或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842358

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842317) |
| stage-b-test-16-npu-a3 | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842318) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842323) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842368) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842375) |
| multimodal-gen-test-1-npu-a3 | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842379) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842392) |
| stage-b-test-1-npu-a2 (1) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842462) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470650322/job/90639842951) |


## [Run #30470508134](https://github.com/sgl-project/sglang/actions/runs/30470508134)
- **分支**: `autotune-extend-buckets`
- **总耗时**: 86.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30470508134

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298406) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298447) |
| stage-b-test-1-npu-a2 (1) | 32.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298464) |
| stage-b-test-1-npu-a2 (0) | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298521) |
| multimodal-gen-test-1-npu-a3 | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298526) |
| multimodal-gen-test-2-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298536) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298547) |
| stage-b-test-4-npu-a3 | 39.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298550) |
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640298686) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470508134/job/90640299239) |


## [Run #30470280443](https://github.com/sgl-project/sglang/actions/runs/30470280443)
- **分支**: `sam/prompt-logprob-fast-topk`
- **总耗时**: 87.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30470280443

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640215931) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640215965) |
| multimodal-gen-test-2-npu-a3 | 46.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640216013) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640216064) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640216074) |
| stage-b-test-1-npu-a2 (0) | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640216088) |
| multimodal-gen-test-1-npu-a3 | 37.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640216109) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640216121) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640216238) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30470280443/job/90640216530) |


## [Run #30461974499](https://github.com/sgl-project/sglang/actions/runs/30461974499)
- **分支**: `main`
- **总耗时**: 39.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30461974499

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 38.3min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165215) |
| stage-b-test-4-npu-a3 | 38.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165232) |
| multimodal-gen-test-1-npu-a3 | 34.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165367) |

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常（进度89%），但随后出现"Executing the custom container implementation failed"错误，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165215

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常，但在15:18:13时出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165232

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165367

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165248) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165334) |
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165358) |
| multimodal-gen-test-2-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165361) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165386) |
| stage-b-test-16-npu-a3 | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165445) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30461974499/job/90610165997) |


## [Run #30460798335](https://github.com/sgl-project/sglang/actions/runs/30460798335)
- **分支**: `dit-full-forward-cuda-graph`
- **总耗时**: 39.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30460798335

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460798335/job/90606135898) |
| multimodal-gen-test-2-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460798335/job/90606135922) |


## [Run #30460773057](https://github.com/sgl-project/sglang/actions/runs/30460773057)
- **分支**: `ulysses-ipc-a2a-2rank`
- **总耗时**: 20.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30460773057

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 19.5min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30460773057/job/90606088111) |
| multimodal-gen-test-2-npu-a3 | 19.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30460773057/job/90606088285) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30460773057/job/90606088111

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体错误，仅有Node.js 20弃用警告和diffusion-failures目录无文件上传的提示，无法判断真实失败原因，可能为日志截断或测试未运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/30460773057/job/90606088285


## [Run #30460488544](https://github.com/sgl-project/sglang/actions/runs/30460488544)
- **分支**: `agent/kernel-inventory-hygiene`
- **总耗时**: 45.1min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30460488544

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 44.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460488544/job/90605105043) |
| stage-b-test-1-npu-a2 (1) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460488544/job/90605105088) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460488544/job/90605105117) |
| stage-b-test-16-npu-a3 | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460488544/job/90605105138) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460488544/job/90605105185) |
| stage-b-test-2-npu-a2 (0) | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460488544/job/90605105294) |
| stage-b-test-4-npu-a3 | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460488544/job/90605105314) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30460488544/job/90605105680) |


## [Run #30459057316](https://github.com/sgl-project/sglang/actions/runs/30459057316)
- **分支**: `main`
- **总耗时**: 32.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30459057316

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 31.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30459057316/job/90600320891) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30459057316/job/90600320891

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30459057316/job/90600320798) |


## [Run #30458575170](https://github.com/sgl-project/sglang/actions/runs/30458575170)
- **分支**: `main`
- **总耗时**: 6.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30458575170

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.3min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传diffusion-failures产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30458575170/job/90598588075) |
| multimodal-gen-test-2-npu-a3 | 5.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30458575170/job/90598588306) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30458575170/job/90598588075

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30458575170/job/90598588306


## [Run #30457884840](https://github.com/sgl-project/sglang/actions/runs/30457884840)
- **分支**: `piotr/lfm2-moe-base-serving`
- **总耗时**: 46.7min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30457884840

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179719) |
| stage-b-test-16-npu-a3 | 19.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179720) |
| stage-b-test-4-npu-a3 | 40.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179728) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179733) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179739) |
| multimodal-gen-test-1-npu-a3 | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179751) |
| stage-b-test-1-npu-a2 (0) | 46.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179781) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179784) |
| multimodal-gen-test-2-npu-a3 | 39.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596179849) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457884840/job/90596180347) |


## [Run #30457407793](https://github.com/sgl-project/sglang/actions/runs/30457407793)
- **分支**: `main`
- **总耗时**: 9.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30457407793

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 8.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627692) |
| stage-b-test-2-npu-a2 (0) | 8.6min | 环境问题 | 自定义容器执行失败，可能是NPU资源或容器环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627724) |
| stage-b-test-2-npu-a2 (1) | 8.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627747) |
| stage-b-test-1-npu-a2 (1) | 8.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627776) |
| stage-b-test-4-npu-a3 | 8.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627789) |
| stage-b-test-1-npu-a2 (0) | 8.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627822) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.7min | 其他 | 作业日志不完整，未显示实际测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594628165) |

- **stage-b-test-16-npu-a3**: 日志显示在模型分片加载过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627692

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行正常，但在执行过程中出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于基础设施或容器环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627724

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行中（Prefill batch正常），但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627747

- **stage-b-test-1-npu-a2 (1)**: 日志显示在运行测试时出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于基础设施或容器环境问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627776

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常，但在Decode阶段后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627789

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但在13:53:33出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于基础设施/容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627822

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中未包含测试运行的关键输出（如性能指标、错误堆栈），仅显示runner初始化、依赖下载和plog备份步骤，无法判断具体失败原因，可能为日志截断或作业在测试前已终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594628165

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30457407793/job/90594627763) |


## [Run #30456622271](https://github.com/sgl-project/sglang/actions/runs/30456622271)
- **分支**: `codex/fix-kimi-trtllm-prefill-graph`
- **总耗时**: 42.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30456622271

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200424) |
| stage-b-test-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200442) |
| stage-a-unit-test-npu | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200479) |
| stage-b-test-1-npu-a2 (1) | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200492) |
| stage-b-test-1-npu-a2 (0) | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200507) |
| multimodal-gen-test-2-npu-a3 | 37.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200520) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200565) |
| multimodal-gen-test-1-npu-a3 | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200587) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592200807) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456622271/job/90592201074) |


## [Run #30456161478](https://github.com/sgl-project/sglang/actions/runs/30456161478)
- **分支**: `codex/mm-encoder-dp-guidance`
- **总耗时**: 43.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30456161478

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381336) |
| stage-a-unit-test-npu | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381338) |
| multimodal-gen-test-1-npu-a3 | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381371) |
| stage-b-test-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381388) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381408) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381417) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381428) |
| stage-b-test-2-npu-a2 (1) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381430) |
| multimodal-gen-test-2-npu-a3 | 40.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590381481) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30456161478/job/90590382042) |


## [Run #30455428845](https://github.com/sgl-project/sglang/actions/runs/30455428845)
- **分支**: `codex/cuda-video-output-finalization`
- **总耗时**: 41.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30455428845

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30455428845/job/90588272568) |
| multimodal-gen-test-2-npu-a3 | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30455428845/job/90588272609) |


---
*Auto-generated by npu_pr_monitor.py*