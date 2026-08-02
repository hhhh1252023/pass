# NPU CI 执行监控
**生成时间**: 2026-08-02 00:05 UTC
**分析 Run 数**: 6

---

## [Run #30720231240](https://github.com/sgl-project/sglang/actions/runs/30720231240)
- **分支**: `main`
- **总耗时**: 78.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30720231240

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 78.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710674) |
| multimodal-gen-test-1-npu-a3 | 78.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710681) |
| stage-b-test-4-npu-a3 | 78.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710688) |
| multimodal-gen-test-2-npu-a3 | 78.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710728) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 78.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710887) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 78.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710899) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 78.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710900) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 78.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710912) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 78.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710923) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 78.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710925) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710674

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710681

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710688

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置问题，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710728

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明作业尝试访问的存储对象已被删除或路径错误，可能是CI配置中引用的工件或依赖文件缺失，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710887

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710899

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的某个 blob 不存在，可能是日志文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710900

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误信息为BlobNotFound，表明作业尝试访问的存储资源缺失或路径错误，属于环境配置或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710912

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 错误码BlobNotFound表明CI作业尝试下载的远程资源（如模型权重或测试数据）已被删除或路径错误，属于基础设施或配置问题，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710923

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及权限。
  链接: https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710925

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710683) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710685) |
| stage-b-test-1-npu-a2 (1) | 32.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710689) |
| stage-b-test-1-npu-a2 (0) | 35.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710690) |
| stage-b-test-2-npu-a2 (1) | 10.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30720231240/job/91422710701) |


## [Run #30717838686](https://github.com/sgl-project/sglang/actions/runs/30717838686)
- **分支**: `main`
- **总耗时**: 67.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30717838686

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 66.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517892) |
| stage-b-test-4-npu-a3 | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517922) |
| stage-b-test-16-npu-a3 | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517929) |
| multimodal-gen-test-2-npu-a3 | 66.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517944) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518291) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518296) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518318) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518333) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 66.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518340) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518356) |

- **multimodal-gen-test-1-npu-a3**: 作业尝试下载或访问一个不存在的 Azure Blob（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于外部存储依赖问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517892

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517922

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517929

- **multimodal-gen-test-2-npu-a3**: 作业失败原因是访问Azure Blob存储时返回BlobNotFound错误，即请求的资源不存在。这可能是由于资源被删除、路径错误或上传未完成，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517944

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518291

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518296

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518318

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518333

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的某个blob（可能是模型权重、测试数据或日志文件）在存储账户中缺失或路径错误，属于环境或资源准备问题，需检查CI配置中的blob路径或上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518340

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416518356

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 11.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517902) |
| stage-b-test-2-npu-a2 (0) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517906) |
| stage-b-test-1-npu-a2 (1) | 32.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517910) |
| stage-b-test-1-npu-a2 (0) | 36.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517930) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30717838686/job/91416517957) |


## [Run #30710677173](https://github.com/sgl-project/sglang/actions/runs/30710677173)
- **分支**: `brayden/disable-bcg-nemotron-h`
- **总耗时**: 193.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30710677173

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 192.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500296) |
| stage-b-test-4-npu-a3 | 192.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500301) |
| multimodal-gen-test-1-npu-a3 | 192.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500304) |
| multimodal-gen-test-2-npu-a3 | 192.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500342) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 192.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500818) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 192.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500842) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 192.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500851) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 192.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500863) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 192.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500882) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 192.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500901) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500296

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500301

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500304

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500342

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或数据在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500818

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误信息为BlobNotFound，说明作业依赖的某个文件或资源在存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500842

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500851

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储账户中缺失，可能是文件被误删、路径错误或上传未完成。建议检查CI配置中的blob路径及上传步骤，确认文件存在且权限正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500863

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500882

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500901

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 36.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500300) |
| stage-b-test-2-npu-a2 (0) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500315) |
| stage-b-test-2-npu-a2 (1) | 10.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500320) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500324) |
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30710677173/job/91397500333) |


## [Run #30708508992](https://github.com/sgl-project/sglang/actions/runs/30708508992)
- **分支**: `main`
- **总耗时**: 157.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30708508992

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 156.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728907) |
| multimodal-gen-test-2-npu-a3 | 156.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728916) |
| stage-b-test-4-npu-a3 | 156.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728928) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 156.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729270) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 156.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729280) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 156.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729295) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 156.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729303) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 156.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729315) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 156.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729319) |

- **stage-b-test-16-npu-a3**: 作业在下载或访问Azure Blob存储中的某个文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728907

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问日志时，Azure Blob 返回 BlobNotFound 错误，说明文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728916

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置和依赖产物。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728928

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到，可能是文件被删除、路径错误或上传失败，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729270

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729280

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729295

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是配置问题或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729303

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题，需检查CI配置中的blob路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729315

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 作业在尝试访问Azure Blob存储中的某个文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391729319

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728893) |
| stage-b-test-1-npu-a2 (0) | 35.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728901) |
| stage-b-test-1-npu-a2 (1) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728904) |
| stage-b-test-2-npu-a2 (1) | 11.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728919) |
| multimodal-gen-test-1-npu-a3 | 26.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728921) |
| stage-b-test-2-npu-a2 (0) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30708508992/job/91391728929) |


## [Run #30704732589](https://github.com/sgl-project/sglang/actions/runs/30704732589)
- **分支**: `feat/ref_aware_kv_buffer`
- **总耗时**: 88.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30704732589

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381857997) |
| stage-b-test-16-npu-a3 | 86.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858002) |
| stage-b-test-4-npu-a3 | 86.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858019) |
| multimodal-gen-test-2-npu-a3 | 86.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858027) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 2.1min | 环境问题 | 作业在启动后立即失败，未执行实际测试，可能因环境或配置问题导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858270) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.5min | 其他 | 作业因缺少metrics.json文件而失败，未生成性能指标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858292) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 1.5min | 环境问题 | 测试未实际运行，缺少metrics.json导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858313) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 86.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858318) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 3.2min | 环境问题 | 作业在启动后立即被清理，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858325) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 2.4min | 其他 | 日志被截断，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858354) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、checkout、upload-artifact等步骤，未显示multimodal-gen-test实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381857997

- **stage-b-test-16-npu-a3**: 作业在尝试访问Azure Blob存储时，因指定的blob不存在（BlobNotFound）而失败。这可能是由于文件未上传、路径错误或存储配置问题，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858002

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858019

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858027

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示作业在准备阶段后直接进入清理，未运行测试脚本，且未生成metrics.json，可能因容器启动失败、资源不足或配置错误导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858270

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示上传artifacts时未找到/tmp/metrics.json，说明性能测试未成功产出结果，可能因测试提前退出或环境问题导致，但无明确错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858292

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示作业在准备阶段即结束，未执行实际测试，且提示找不到/tmp/metrics.json，无法上传性能指标，属于环境或配置问题导致测试未启动。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858313

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个数据文件或模型权重在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858318

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示作业在初始化阶段（约3分钟内）即被终止，仅包含GitHub Actions环境准备和清理信息，未出现任何测试执行或性能数据，疑似runner被提前回收或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858325

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志仅包含GitHub Actions初始化、依赖下载及作业清理步骤，未展示实际测试运行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858354

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 35.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858010) |
| stage-a-unit-test-npu | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858011) |
| stage-b-test-2-npu-a2 (1) | 10.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858020) |
| stage-b-test-2-npu-a2 (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858021) |
| stage-b-test-1-npu-a2 (1) | 32.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30704732589/job/91381858026) |


## [Run #30702820485](https://github.com/sgl-project/sglang/actions/runs/30702820485)
- **分支**: `xinyuan/anthropic-tool-is-error`
- **总耗时**: 600.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30702820485

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 28.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755117) |
| stage-b-test-4-npu-a3 | 18.9min | 代码错误 | NPU测试中HiCache MLA测试失败，导致作业整体失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755196) |
| single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms | 27.9min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91387920830) |
| single-node-poc (qwen3_5_9b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/qw... / qwen3_5_9b_bf16_1p_gsm8k | 34.6min | 其他 | 日志被截断，未显示测试执行结果，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91398882968) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755117

- **stage-b-test-4-npu-a3**: 测试test_npu_hicache_mla.py返回退出码1，执行超时（418秒，预估400秒），其余测试通过。可能是该测试用例存在逻辑错误或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755196

- **single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms**: 日志仅包含作业初始化和清理阶段，中间部分被省略，未展示测试运行、错误信息或退出码，因此无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91387920830

- **single-node-poc (qwen3_5_9b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/qw... / qwen3_5_9b_bf16_1p_gsm8k**: 日志仅包含作业启动和清理信息，中间部分被省略，未展示测试命令输出或错误信息，因此无法确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91398882968

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755105) |
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755110) |
| stage-b-test-1-npu-a2 (0) | 35.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755111) |
| stage-b-test-1-npu-a2 (1) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755112) |
| stage-b-test-2-npu-a2 (1) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755113) |
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755119) |
| multimodal-gen-test-1-npu-a3 | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755180) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755403) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755432) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755438) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 50.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755455) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755458) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91376755476) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91383389401) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91383676967) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91383869831) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91384055573) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91384446946) |
| single-node-poc (qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu... / qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91393690779) |
| single-node-poc (glm4_7_flash_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/glm4_... / glm4_7_flash_1p_gsm8k | 49.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91394295933) |
| single-node-poc (qwen3_vl_30b_a3b_bf16_2p_gsm8k, linux-aarch64-a3-4, test/registered/ascend/accur... / qwen3_vl_30b_a3b_bf16_2p_gsm8k | 44.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91394494774) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16, test/registered/ascend/acc... / glm5_top64_pruned_bf16_8p_gsm8k | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91395808695) |
| single-node-poc (moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/... / moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30702820485/job/91395852841) |


---
*Auto-generated by npu_pr_monitor.py*