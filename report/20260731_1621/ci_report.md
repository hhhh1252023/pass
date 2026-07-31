# NPU CI 执行监控
**生成时间**: 2026-07-31 08:21 UTC
**分析 Run 数**: 24

---

## [Run #30140986579](https://github.com/sgl-project/sglang/actions/runs/30140986579)
- **分支**: `ut`
- **总耗时**: 44.2min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30140986579

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099864) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099865) |
| multimodal-gen-test-1-npu-a3 | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099867) |
| stage-b-test-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099871) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099877) |
| stage-b-test-1-npu-a2 (0) | 43.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099889) |
| stage-b-test-1-npu-a2 (1) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099893) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099894) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636099897) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140986579/job/89636100006) |


## [Run #30140674142](https://github.com/sgl-project/sglang/actions/runs/30140674142)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 44.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30140674142

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633257795) |
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633257801) |
| multimodal-gen-test-2-npu-a3 | 34.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633257802) |
| stage-b-test-1-npu-a2 (0) | 42.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633257806) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633257812) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633257819) |
| stage-b-test-2-npu-a2 (0) | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633257825) |
| stage-b-test-4-npu-a3 | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633257856) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140674142/job/89633258049) |


## [Run #30140511339](https://github.com/sgl-project/sglang/actions/runs/30140511339)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 5.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30140511339

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769736) |
| stage-b-test-1-npu-a2 (1) | 4.3min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769743) |
| stage-b-test-1-npu-a2 (0) | 4.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769745) |
| stage-b-test-4-npu-a3 | 3.8min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769751) |
| stage-b-test-2-npu-a2 (0) | 4.3min | 环境问题 | 自定义容器执行失败，导致测试未运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769752) |
| multimodal-gen-test-2-npu-a3 | 4.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769753) |
| stage-b-test-16-npu-a3 | 4.7min | 环境问题 | 自定义容器执行失败，NPU作业在模型加载阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769755) |
| stage-b-test-2-npu-a2 (1) | 3.9min | 环境问题 | 下载triton-ascend依赖时自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769756) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.1min | 环境问题 | 测试未生成metrics.json，导致性能测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769943) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、checkout、upload-artifact等步骤，未展示multimodal-gen-test的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769736

- **stage-b-test-1-npu-a2 (1)**: 日志显示在运行测试命令后，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU驱动、容器资源或环境配置异常，导致测试无法启动。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769743

- **stage-b-test-1-npu-a2 (0)**: 作业在启动第一个NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769745

- **stage-b-test-4-npu-a3**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是容器环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769751

- **stage-b-test-2-npu-a2 (0)**: 日志显示在运行测试命令后，自定义容器实现执行失败（Executing the custom container implementation failed），可能是NPU环境或容器配置问题，需联系自托管runner管理员。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769752

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败产物，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769753

- **stage-b-test-16-npu-a3**: 日志显示作业在加载模型分片（约9/161）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769755

- **stage-b-test-2-npu-a2 (1)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包后，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769756

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业在运行性能测试后未找到/tmp/metrics.json文件，无法上传性能指标，可能因测试环境或脚本问题导致性能数据未生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140511339/job/89632769943


## [Run #30140062238](https://github.com/sgl-project/sglang/actions/runs/30140062238)
- **分支**: `mmangkad/torch-2.12`
- **总耗时**: 48.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30140062238

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631944926) |
| multimodal-gen-test-1-npu-a3 | 36.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631944928) |
| multimodal-gen-test-2-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631944930) |
| stage-b-test-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631944933) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631944936) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631944941) |
| stage-b-test-1-npu-a2 (1) | 31.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631944955) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631944964) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30140062238/job/89631945155) |


## [Run #30140034295](https://github.com/sgl-project/sglang/actions/runs/30140034295)
- **分支**: `mmangkad/torch-2.12`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30140034295

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 0.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140034295/job/89631434933) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30140034295/job/89631435026) |

- **stage-b-test-1-npu-a2 (0)**: 作业失败是因为访问 Azure Blob 时返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140034295/job/89631434933

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30140034295/job/89631435026


## [Run #30139831807](https://github.com/sgl-project/sglang/actions/runs/30139831807)
- **分支**: `fangyuan/dspark`
- **总耗时**: 44.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30139831807

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877227) |
| stage-b-test-16-npu-a3 | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877228) |
| stage-b-test-1-npu-a2 (1) | 32.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877230) |
| stage-b-test-1-npu-a2 (0) | 42.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877232) |
| multimodal-gen-test-1-npu-a3 | 35.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877236) |
| stage-b-test-4-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877241) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877253) |
| multimodal-gen-test-2-npu-a3 | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877257) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139831807/job/89630877365) |


## [Run #30139466431](https://github.com/sgl-project/sglang/actions/runs/30139466431)
- **分支**: `main`
- **总耗时**: 42.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30139466431

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139466431/job/89629840453) |
| multimodal-gen-test-1-npu-a3 | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139466431/job/89629840455) |


## [Run #30139233628](https://github.com/sgl-project/sglang/actions/runs/30139233628)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 40.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30139233628

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 39.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226206) |
| multimodal-gen-test-2-npu-a3 | 38.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226231) |
| multimodal-gen-test-1-npu-a3 | 38.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226236) |
| stage-b-test-1-npu-a2 (1) | 38.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226262) |
| stage-b-test-1-npu-a2 (0) | 39.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226269) |

- **stage-b-test-4-npu-a3**: 日志显示Prefill正常进行，但随后自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器运行问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226206

- **multimodal-gen-test-2-npu-a3**: 日志中间部分省略，仅显示上传diffusion-failures目录时无文件，无法判断具体失败原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226231

- **multimodal-gen-test-1-npu-a3**: 日志中只有checkout、upload-artifact等常规步骤，未包含multimodal测试执行的具体输出或错误信息，无法判断失败原因，可能为日志截断或作业在测试前已异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226236

- **stage-b-test-1-npu-a2 (1)**: 日志显示在apt更新过程中长时间无响应，最终报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226262

- **stage-b-test-1-npu-a2 (0)**: 作业在启动TokenizerManager后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226269

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226232) |
| stage-b-test-2-npu-a2 (0) | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226255) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226275) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30139233628/job/89629226453) |


## [Run #30138462945](https://github.com/sgl-project/sglang/actions/runs/30138462945)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 23.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30138462945

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 19.9min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103444) |
| multimodal-gen-test-2-npu-a3 | 21.3min | 其他 | 作业失败但日志未显示明确错误，仅上传失败产物时提示无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103445) |
| stage-b-test-4-npu-a3 | 21.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103467) |
| stage-b-test-2-npu-a2 (1) | 21.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103479) |
| stage-b-test-1-npu-a2 (1) | 21.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103481) |
| stage-b-test-1-npu-a2 (0) | 21.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103504) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业正常执行，上传diffusion-failures目录时提示无文件，未出现测试失败或错误信息，可能为作业提前结束或测试未产生失败产物。
  链接: https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103444

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或提前退出，需查看完整日志定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103445

- **stage-b-test-4-npu-a3**: 日志显示在测试运行过程中，自定义容器实现执行失败，提示联系自托管 runner 管理员，属于基础设施或容器环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103467

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103479

- **stage-b-test-1-npu-a2 (1)**: 作业在加载模型权重后执行自定义容器时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103481

- **stage-b-test-1-npu-a2 (0)**: 测试运行到第二个用例时，自定义容器实现执行失败，导致作业提前终止。第一个用例正常通过，但环境在第二个测试开始时崩溃，属于NPU容器环境不稳定问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103504

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103448) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103455) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30138462945/job/89627103599) |


## [Run #30137580167](https://github.com/sgl-project/sglang/actions/runs/30137580167)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 26.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30137580167

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 25.3min | 环境问题 | 自托管runner执行自定义容器实现时失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571216) |
| stage-b-test-2-npu-a2 (0) | 21.0min | 环境问题 | 自定义容器执行失败，apt源下载超时导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571222) |
| multimodal-gen-test-2-npu-a3 | 24.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571224) |
| multimodal-gen-test-1-npu-a3 | 24.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571226) |
| stage-b-test-2-npu-a2 (1) | 20.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571234) |
| stage-b-test-1-npu-a2 (0) | 20.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571246) |
| stage-b-test-1-npu-a2 (1) | 24.2min | 超时 | Scheduler watchdog 超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571248) |

- **stage-b-test-4-npu-a3**: 日志显示测试运行到99%时，runner报错“Executing the custom container implementation failed”，随后进入清理流程。这属于runner环境或容器执行问题，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571216

- **stage-b-test-2-npu-a2 (0)**: 作业在apt更新阶段卡住约19分钟（01:06:33至01:25:44），最终报错“Executing the custom container implementation failed”，疑似自托管runner环境或网络问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571222

- **multimodal-gen-test-2-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571224

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、checkout、上传artifact等步骤，未显示multimodal-gen测试的具体执行和失败输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571226

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行中（Prefill batch正常），但突然报错“Executing the custom container implementation failed”，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571234

- **stage-b-test-1-npu-a2 (0)**: 日志显示NPU图捕获正常完成，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571246

- **stage-b-test-1-npu-a2 (1)**: 日志显示 Scheduler watchdog timeout (self.watchdog_timeout=300)，批处理捕获进度缓慢，最终触发超时，作业被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571248

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571230) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30137580167/job/89624571423) |


## [Run #30137001307](https://github.com/sgl-project/sglang/actions/runs/30137001307)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 16.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30137001307

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 10.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917916) |
| multimodal-gen-test-2-npu-a3 | 9.9min | 环境问题 | 作业因缺少diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917922) |
| stage-b-test-16-npu-a3 | 10.2min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917928) |
| stage-b-test-2-npu-a2 (1) | 12.4min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917930) |
| multimodal-gen-test-1-npu-a3 | 13.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917933) |
| stage-b-test-2-npu-a2 (0) | 12.0min | 环境问题 | 自定义容器执行失败，NPU分布式初始化未完成 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917941) |
| stage-b-test-1-npu-a2 (0) | 14.4min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917943) |
| stage-b-test-1-npu-a2 (1) | 12.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917958) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.1min | 其他 | 作业日志被截断，未显示实际失败原因，仅看到清理和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622918246) |

- **stage-b-test-4-npu-a3**: 作业在启动NPU推理服务时，自定义容器实现执行失败，导致测试无法继续。日志显示服务初始化正常，但容器运行环境出现问题，可能是NPU资源或容器配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917916

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures工件时未找到文件，说明测试未生成失败样本，可能因环境配置或前置步骤异常导致测试未运行，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917922

- **stage-b-test-16-npu-a3**: 日志显示模型分片加载到91%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题，非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917928

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常，但中途出现 'Executing the custom container implementation failed' 错误，属于自托管 runner 环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917930

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体输出，仅有checkout、upload-artifact等步骤，且upload-artifact提示无文件上传，可能测试未运行或日志被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917933

- **stage-b-test-2-npu-a2 (0)**: 作业在TP1/TP0初始化torch distributed时，自定义容器实现执行失败，导致作业中断。可能是NPU环境配置或容器资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917941

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但在01:00:25时出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917943

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常，但在01:00:25时出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622917958

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分省略，末尾仅显示plog备份、Node警告和清理流程，未出现测试执行结果或错误信息，无法判断具体失败点，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30137001307/job/89622918246


## [Run #30136979342](https://github.com/sgl-project/sglang/actions/runs/30136979342)
- **分支**: `main`
- **总耗时**: 45.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30136979342

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831417) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831424) |
| stage-b-test-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831442) |
| stage-b-test-16-npu-a3 | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831450) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831459) |
| multimodal-gen-test-2-npu-a3 | 43.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831460) |
| stage-b-test-1-npu-a2 (1) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831504) |
| stage-b-test-1-npu-a2 (0) | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831512) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136979342/job/89622831666) |


## [Run #30136690543](https://github.com/sgl-project/sglang/actions/runs/30136690543)
- **分支**: `fix_aiter_preshuffle_mqa`
- **总耗时**: 47.7min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30136690543

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952494) |
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952499) |
| stage-b-test-1-npu-a2 (1) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952512) |
| multimodal-gen-test-1-npu-a3 | 31.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952513) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952519) |
| multimodal-gen-test-2-npu-a3 | 37.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952523) |
| stage-b-test-2-npu-a2 (1) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952528) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952553) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30136690543/job/89621952834) |


## [Run #30136515876](https://github.com/sgl-project/sglang/actions/runs/30136515876)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 13.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30136515876

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 11.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434294) |
| stage-b-test-16-npu-a3 | 10.1min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434316) |
| stage-b-test-2-npu-a2 (1) | 10.1min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434319) |
| multimodal-gen-test-1-npu-a3 | 12.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434323) |
| stage-b-test-2-npu-a2 (0) | 0.9min | 环境问题 | 自托管runner执行自定义容器实现失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434328) |
| stage-b-test-4-npu-a3 | 10.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434334) |
| stage-b-test-1-npu-a2 (1) | 9.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434335) |
| stage-b-test-1-npu-a2 (0) | 2.4min | 环境问题 | pip安装依赖时网络超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434349) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.7min | 其他 | 日志被截断，无法定位具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434654) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434294

- **stage-b-test-16-npu-a3**: 作业在加载模型分片至87%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434316

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常，但在进度34%时出现'Executing the custom container implementation failed'错误，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434319

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions基础设施的警告（如Node 20弃用）和上传artifact时无文件的通知，无法判断测试失败的具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434323

- **stage-b-test-2-npu-a2 (0)**: 作业在apt更新阶段后报错"Executing the custom container implementation failed"，属于runner环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434328

- **stage-b-test-4-npu-a3**: 日志显示测试运行到75%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434334

- **stage-b-test-1-npu-a2 (1)**: 日志显示在TokenizerManager初始化后，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于运行环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434335

- **stage-b-test-1-npu-a2 (0)**: 在安装triton-ascend时，访问内部pypi缓存服务（cache-service.nginx-pypi-cache.svc.cluster.local）出现Read timed out，导致依赖下载失败，作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434349

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志仅包含作业初始化和清理阶段，未显示测试执行及失败关键信息，无法判断是性能、精度还是环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136515876/job/89621434654


## [Run #30136482740](https://github.com/sgl-project/sglang/actions/runs/30136482740)
- **分支**: `main`
- **总耗时**: 13.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30136482740

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 12.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319215) |
| stage-b-test-4-npu-a3 | 11.5min | 环境问题 | 自定义容器执行失败，模型权重加载过程中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319217) |
| stage-b-test-2-npu-a2 (0) | 12.1min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319228) |
| stage-b-test-16-npu-a3 | 12.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319235) |
| stage-b-test-2-npu-a2 (1) | 10.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319237) |
| multimodal-gen-test-1-npu-a3 | 11.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319242) |
| stage-b-test-1-npu-a2 (0) | 9.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319266) |
| stage-b-test-1-npu-a2 (1) | 10.2min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319272) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.4min | 其他 | 日志被截断，未显示测试执行失败的具体原因，仅看到作业结束和清理过程。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319387) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node.js版本弃用警告和上传artifact时无文件提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319215

- **stage-b-test-4-npu-a3**: 作业在加载模型权重（约19%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，可能因NPU环境或容器配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319217

- **stage-b-test-2-npu-a2 (0)**: 作业在运行第二个测试时，自定义容器实现执行失败，导致测试进程被终止。日志显示第一个测试已通过，但第二个测试刚开始即报错，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319228

- **stage-b-test-16-npu-a3**: 日志显示在TP/EP初始化后，出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319235

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试正常运行至40%时，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319237

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含多模态生成测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319242

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行至92%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319266

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常（Prefill batch处理中），但随后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319272

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分被省略，无法定位失败点。作业在运行后直接进入清理阶段，未输出测试结果或错误信息，可能因日志不完整导致无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30136482740/job/89621319387


## [Run #30135915721](https://github.com/sgl-project/sglang/actions/runs/30135915721)
- **分支**: `cheng/presharded-load-format`
- **总耗时**: 243.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30135915721

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (1) | 242.1min | 环境问题 | Python 环境缺少 tabulate 模块导致测试脚本无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634014) |

- **stage-b-test-2-npu-a2 (1)**: run_suite.py 导入 tabulate 失败，说明 CI 环境的 Python 依赖未正确安装或未激活对应虚拟环境，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634014

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634009) |
| multimodal-gen-test-1-npu-a3 | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634011) |
| stage-b-test-16-npu-a3 | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634015) |
| stage-b-test-1-npu-a2 (0) | 42.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634016) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634025) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634036) |
| multimodal-gen-test-2-npu-a3 | 47.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634044) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135915721/job/89619634274) |


## [Run #30135369474](https://github.com/sgl-project/sglang/actions/runs/30135369474)
- **分支**: `main`
- **总耗时**: 14.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30135369474

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 13.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076963) |
| stage-b-test-4-npu-a3 | 13.0min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076964) |
| multimodal-gen-test-2-npu-a3 | 13.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076971) |
| stage-b-test-1-npu-a2 (0) | 13.3min | 环境问题 | NPU测试执行过程中自定义容器实现失败，导致作业中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076975) |
| multimodal-gen-test-1-npu-a3 | 12.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076978) |
| stage-b-test-2-npu-a2 (0) | 13.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076979) |
| stage-b-test-1-npu-a2 (1) | 13.3min | 环境问题 | 自定义容器执行失败，作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618077000) |
| stage-b-test-2-npu-a2 (1) | 13.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618077025) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 12.9min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618077163) |

- **stage-b-test-16-npu-a3**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076963

- **stage-b-test-4-npu-a3**: 日志显示在加载模型权重到56%时，runner报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076964

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node版本弃用警告和上传artifact时无文件提示，无法判断失败根因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076971

- **stage-b-test-1-npu-a2 (0)**: 日志显示在Capturing batches阶段（83%）后，出现错误：Executing the custom container implementation failed，提示联系self hosted runner管理员。这属于运行环境或容器问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076975

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node.js版本弃用警告和上传artifact时无文件提示，无法判断具体失败点，可能为日志截断或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076978

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行到37%时，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于NPU环境或容器执行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618076979

- **stage-b-test-1-npu-a2 (1)**: 日志显示第一个测试通过后，在开始第二个测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618077000

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试进行到81%时，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618077025

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中未包含测试运行的具体输出或错误信息，仅显示作业开始、输入配置、plog备份和清理步骤，无法判断失败原因，可能因日志截断或作业在早期阶段被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30135369474/job/89618077163


## [Run #30135161922](https://github.com/sgl-project/sglang/actions/runs/30135161922)
- **分支**: `hicache-shm-allocator`
- **总耗时**: 43.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30135161922

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483402) |
| multimodal-gen-test-2-npu-a3 | 33.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483403) |
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483410) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483414) |
| stage-b-test-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483423) |
| stage-b-test-1-npu-a2 (0) | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483434) |
| stage-b-test-2-npu-a2 (1) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483447) |
| stage-b-test-16-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483457) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30135161922/job/89617483539) |


## [Run #30133809921](https://github.com/sgl-project/sglang/actions/runs/30133809921)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 42.6min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30133809921

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628154) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628157) |
| stage-b-test-2-npu-a2 (1) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628163) |
| stage-b-test-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628165) |
| multimodal-gen-test-1-npu-a3 | 33.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628168) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628187) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628198) |
| multimodal-gen-test-2-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628204) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133809921/job/89613628515) |


## [Run #30133463912](https://github.com/sgl-project/sglang/actions/runs/30133463912)
- **分支**: `main`
- **总耗时**: 42.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30133463912

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 35.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594420) |
| multimodal-gen-test-2-npu-a3 | 40.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594432) |
| stage-b-test-1-npu-a2 (0) | 41.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594463) |

- **stage-b-test-4-npu-a3**: 日志显示Prefill正常进行，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594420

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误或失败断言，仅显示Node.js 20弃用警告和上传artifact时无文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594432

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但突然出现“Executing the custom container implementation failed”错误，可能是自托管runner的容器环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594463

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594413) |
| multimodal-gen-test-1-npu-a3 | 27.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594422) |
| stage-b-test-2-npu-a2 (0) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594428) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594429) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594437) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30133463912/job/89612594787) |


## [Run #30132677201](https://github.com/sgl-project/sglang/actions/runs/30132677201)
- **分支**: `main`
- **总耗时**: 17.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30132677201

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 15.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378570) |
| multimodal-gen-test-1-npu-a3 | 15.8min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378591) |
| stage-b-test-4-npu-a3 | 15.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378602) |
| stage-b-test-2-npu-a2 (1) | 15.6min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378608) |
| stage-b-test-2-npu-a2 (0) | 15.8min | 环境问题 | 测试全部通过但作业失败，原因是自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378609) |
| stage-b-test-1-npu-a2 (0) | 15.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378616) |
| stage-b-test-1-npu-a2 (1) | 15.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378628) |
| multimodal-gen-test-2-npu-a3 | 15.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378651) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.4min | 其他 | 作业日志被截断，未显示实际失败原因，仅见清理和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378941) |

- **stage-b-test-16-npu-a3**: 日志显示测试运行正常，但在执行过程中出现错误："Executing the custom container implementation failed"，提示联系自托管 runner 管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378570

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但根本原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378591

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常（HTTP 200），但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378602

- **stage-b-test-2-npu-a2 (1)**: 测试运行到第二个用例时，自定义容器实现执行失败，报错提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378608

- **stage-b-test-2-npu-a2 (0)**: 日志显示2/2测试全部通过，但随后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378609

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行至18%时，自定义容器实现执行失败，提示联系自托管runner管理员，可能因NPU资源或容器环境不稳定导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378616

- **stage-b-test-1-npu-a2 (1)**: 日志显示在批量捕获完成后，出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378628

- **multimodal-gen-test-2-npu-a3**: 作业在运行multimodal-gen测试后上传diffusion-failures目录时提示无文件，但关键测试输出被省略，无法判断具体失败点，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378651

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中间部分省略，末尾仅显示plog备份、节点清理及Node.js弃用警告，未捕获测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132677201/job/89610378941


## [Run #30132200194](https://github.com/sgl-project/sglang/actions/runs/30132200194)
- **分支**: `main`
- **总耗时**: 10.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30132200194

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 7.0min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919738) |
| stage-b-test-16-npu-a3 | 9.2min | 环境问题 | NPU 环境加载模型权重时出现 libtorch 相关错误，导致容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919739) |
| multimodal-gen-test-1-npu-a3 | 6.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919742) |
| stage-b-test-1-npu-a2 (0) | 6.9min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919746) |
| multimodal-gen-test-2-npu-a3 | 1.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919755) |
| stage-b-test-2-npu-a2 (0) | 9.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919780) |
| stage-b-test-4-npu-a3 | 1.5min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919809) |
| stage-b-test-2-npu-a2 (1) | 9.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919814) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 1.8min | 环境问题 | 作业在启动阶段即失败，未执行实际测试，可能因运行环境或资源问题导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608920089) |

- **stage-b-test-1-npu-a2 (1)**: 日志显示torch_npu导入时出现ImportWarning和RuntimeWarning，随后自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU环境配置或兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919738

- **stage-b-test-16-npu-a3**: 日志显示在加载 MoE 模型权重时，libtorch_python.so 抛出 variable_copy_ 相关异常，同时伴随 Scheduler watchdog 超时，最终自定义容器执行失败，属于 NPU 环境或依赖兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919739

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919742

- **stage-b-test-1-npu-a2 (0)**: 日志显示在apt更新过程中，执行自定义容器实现时出错，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919746

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919755

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行正常，但在执行过程中出现错误：'Executing the custom container implementation failed. Please contact your self hosted runner administrator.'，随后作业被终止，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919780

- **stage-b-test-4-npu-a3**: 日志显示在安装Rust组件时，runner报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919809

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行到11%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608919814

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在准备阶段后立即进入清理流程，未运行测试脚本，且无错误信息，可能因runner环境异常或资源分配失败导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30132200194/job/89608920089


## [Run #30130640922](https://github.com/sgl-project/sglang/actions/runs/30130640922)
- **分支**: `jamesl/inkling-dflash-main`
- **总耗时**: 43.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30130640922

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194451) |
| stage-b-test-1-npu-a2 (1) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194453) |
| stage-b-test-4-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194457) |
| multimodal-gen-test-1-npu-a3 | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194458) |
| stage-b-test-16-npu-a3 | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194472) |
| stage-b-test-2-npu-a2 (0) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194475) |
| stage-b-test-2-npu-a2 (1) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194476) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194484) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30130640922/job/89604194700) |


## [Run #30129548767](https://github.com/sgl-project/sglang/actions/runs/30129548767)
- **分支**: `feat/triton-sparse-mla`
- **总耗时**: 43.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30129548767

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 32.4min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865571) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865571

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865512) |
| multimodal-gen-test-2-npu-a3 | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865514) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865537) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865543) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865550) |
| stage-b-test-2-npu-a2 (0) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865557) |
| stage-b-test-4-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865712) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30129548767/job/89600865799) |


---
*Auto-generated by npu_pr_monitor.py*