# NPU CI 执行监控
**生成时间**: 2026-08-07 08:12 UTC
**分析 Run 数**: 21

---

## [Run #31154674076](https://github.com/sgl-project/sglang/actions/runs/31154674076)
- **分支**: `lsyin/ci-trim-h100`
- **总耗时**: 70.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31154674076

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 69.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684209) |
| stage-b-test-4-npu-a3 (1) | 69.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684226) |
| stage-b-test-16-npu-a3 | 69.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684231) |
| stage-b-test-4-npu-a3 (0) | 69.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684255) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 69.4min | 环境问题 | Azure Blob 存储中指定的模型文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684750) |

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684209

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的构建产物或缓存文件未上传或已被删除，属于基础设施/环境配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684226

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684231

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684255

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明模型权重或数据文件在存储中缺失或路径错误，可能是上传失败或配置引用错误，需检查存储路径及文件完整性。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684750

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684212) |
| stage-b-test-1-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684222) |
| stage-b-test-2-npu-a3 | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31154674076/job/92791684295) |


## [Run #31154656979](https://github.com/sgl-project/sglang/actions/runs/31154656979)
- **分支**: `lsyin/swa-sync-free`
- **总耗时**: 14.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31154656979

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534914) |
| stage-b-test-8-npu-a3 | 13.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534932) |
| stage-b-test-4-npu-a3 (1) | 13.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534935) |
| multimodal-gen-test-2-npu-a3 | 13.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534951) |
| stage-b-test-1-npu-a3 | 13.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534957) |
| stage-b-test-2-npu-a3 | 13.7min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534963) |
| stage-b-test-4-npu-a3 (0) | 13.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534965) |
| stage-b-test-16-npu-a3 | 13.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534988) |
| stage-a-unit-test-npu | 6.1min | 环境问题 | rustup 下载中断导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534992) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 13.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791535344) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 13.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791535350) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534914

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534932

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534935

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534951

- **stage-b-test-1-npu-a3**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534957

- **stage-b-test-2-npu-a3**: 作业在加载模型权重（约31%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534963

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534965

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于环境或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534988

- **stage-a-unit-test-npu**: CI 在安装 Rust 时，从内部缓存服务下载 rustup-init 过程中连接中断（curl 错误 18），文件下载不完整，导致脚本退出码 18，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791534992

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791535344

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明作业依赖的远程存储对象缺失，可能是文件被删除、路径错误或上传未完成。建议检查CI配置中的blob路径及上传流程，确认文件存在且权限正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154656979/job/92791535350


## [Run #31154106234](https://github.com/sgl-project/sglang/actions/runs/31154106234)
- **分支**: `lsyin/swa-sync-free`
- **总耗时**: 9.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31154106234

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838585) |
| stage-b-test-8-npu-a3 | 8.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838616) |
| stage-b-test-1-npu-a3 | 8.9min | 环境问题 | 自定义容器执行失败，模型权重加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838632) |
| stage-b-test-16-npu-a3 | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838638) |
| stage-b-test-4-npu-a3 (1) | 4.1min | 环境问题 | 自定义容器执行失败，模型加载权重时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838646) |
| multimodal-gen-test-2-npu-a3 | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838653) |
| stage-b-test-2-npu-a3 | 8.9min | 环境问题 | NPU后端不支持cuda设备类型，导致SymmetricMemory功能禁用，且容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838658) |
| stage-b-test-4-npu-a3 (0) | 5.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838674) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 8.8min | 环境问题 | Azure Blob 存储中指定的模型权重文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838916) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838917) |

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问某个blob时，返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置问题，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838585

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838616

- **stage-b-test-1-npu-a3**: 作业在加载模型权重（25%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838632

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于环境或资源配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838638

- **stage-b-test-4-npu-a3 (1)**: 作业在加载模型权重（Multi-thread loading shards 0%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838646

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838653

- **stage-b-test-2-npu-a3**: 日志显示multimem all-gather disabled (SymmetricMemory does not support device type cuda)，说明NPU环境与CUDA相关功能不兼容，最终自定义容器执行失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838658

- **stage-b-test-4-npu-a3 (0)**: 日志显示在测试运行过程中，自定义容器实现执行失败（Executing the custom container implementation failed），提示联系自托管runner管理员，属于基础设施环境问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838674

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 glm5_top64_pruned_bf16 模型权重在存储中缺失或路径错误，可能是上传失败或配置错误，需检查模型文件路径及存储状态。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838916

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或测试数据未上传到指定存储路径，或路径配置错误。需检查存储账户中的 blob 是否存在，并确认 CI 配置中的路径是否正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838917

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31154106234/job/92789838642) |


## [Run #31153774839](https://github.com/sgl-project/sglang/actions/runs/31153774839)
- **分支**: `mxfp4-marlin-sm80-gate`
- **总耗时**: 10.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31153774839

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 6.5min | 环境问题 | NPU容器执行失败，自定义容器实现报错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841810) |
| stage-b-test-16-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841876) |
| stage-b-test-8-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841879) |
| stage-b-test-1-npu-a3 | 6.3min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841905) |
| stage-b-test-4-npu-a3 (0) | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841910) |
| stage-b-test-4-npu-a3 (1) | 9.0min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841929) |
| multimodal-gen-test-2-npu-a3 | 6.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841931) |
| multimodal-gen-test-1-npu-a3 | 7.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841939) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788842343) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.7min | 其他 | 日志被截断，无法看到实际失败原因，仅显示作业正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788842360) |

- **stage-b-test-2-npu-a3**: 作业在加载模型权重后，自定义容器执行失败，报错'Executing the custom container implementation failed'，属于NPU环境或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841810

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径和权限。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841876

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841879

- **stage-b-test-1-npu-a3**: 作业在加载模型权重完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841905

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841910

- **stage-b-test-4-npu-a3 (1)**: 日志显示BlobNotFound错误，说明CI作业尝试下载的依赖文件或缓存未在指定存储位置找到，可能是资源被清理、路径错误或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841929

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的具体错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841931

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841939

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明作业依赖的远程数据文件已被删除或路径错误，需检查 CI 配置中的数据引用或重新上传数据。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788842343

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志片段仅包含作业启动和清理过程，未包含测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788842360

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31153774839/job/92788841895) |


## [Run #31153589547](https://github.com/sgl-project/sglang/actions/runs/31153589547)
- **分支**: `main`
- **总耗时**: 30.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31153589547

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 29.2min | 其他 | 作业未显示明确失败原因，仅上传工件时未找到文件，可能测试未执行或提前退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153589547/job/92788321452) |
| multimodal-gen-test-2-npu-a3 | 28.1min | 其他 | 作业未显示明确失败原因，日志仅包含正常执行和Node 20弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153589547/job/92788321521) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时无文件，说明测试可能未运行或未产生失败样本，需进一步查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153589547/job/92788321452

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败、超时或错误信息，仅显示上传artifact时未找到文件（diffusion-failures/），以及Node 20弃用警告。作业可能因测试未运行或结果未生成而失败，需查看更完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153589547/job/92788321521


## [Run #31153376440](https://github.com/sgl-project/sglang/actions/runs/31153376440)
- **分支**: `lsyin/ci-trim-h100`
- **总耗时**: 22.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31153376440

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (1) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597891) |
| stage-b-test-2-npu-a3 | 21.8min | 其他 | 作业日志显示所有测试均通过，无失败迹象。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597895) |
| stage-b-test-16-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597921) |
| stage-b-test-4-npu-a3 (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597928) |
| stage-b-test-1-npu-a3 | 12.8min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597930) |
| stage-b-test-8-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597931) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787598310) |

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597891

- **stage-b-test-2-npu-a3**: 日志中所有NPU测试均显示passed，作业正常完成，可能是误报或日志截断导致。建议检查作业状态是否为成功。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597895

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597921

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597928

- **stage-b-test-1-npu-a3**: 作业在运行第4个测试时，自定义容器实现执行失败，导致作业提前终止。日志显示前3个测试均通过，失败发生在容器层面而非测试代码本身，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597930

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597931

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传或已被删除，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787598310

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787597955) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31153376440/job/92787598305) |


## [Run #31152981859](https://github.com/sgl-project/sglang/actions/runs/31152981859)
- **分支**: `feat/lingbot-video-moe-30b`
- **总耗时**: 28.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31152981859

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 27.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152981859/job/92786402395) |

- **multimodal-gen-test-2-npu-a3**: 日志截断，缺少核心测试执行部分。仅显示上传diffusion-failures工件时未找到文件，说明测试可能未产生失败样本，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152981859/job/92786402395

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31152981859/job/92786402427) |


## [Run #31152969013](https://github.com/sgl-project/sglang/actions/runs/31152969013)
- **分支**: `mick/qwen-host-varlen-meta`
- **总耗时**: 26.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31152969013

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 10.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152969013/job/92786360037) |

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152969013/job/92786360037

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31152969013/job/92786360057) |


## [Run #31152801464](https://github.com/sgl-project/sglang/actions/runs/31152801464)
- **分支**: `fix-diffusion-health-warmup-gate`
- **总耗时**: 30.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31152801464

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 29.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152801464/job/92785871397) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152801464/job/92785871397

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31152801464/job/92785871394) |


## [Run #31152686560](https://github.com/sgl-project/sglang/actions/runs/31152686560)
- **分支**: `main`
- **总耗时**: 16.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31152686560

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 15.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152686560/job/92785525200) |
| multimodal-gen-test-1-npu-a3 | 15.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152686560/job/92785525253) |

- **multimodal-gen-test-2-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152686560/job/92785525200

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未包含multimodal-gen测试的具体执行步骤或错误输出，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152686560/job/92785525253


## [Run #31152370081](https://github.com/sgl-project/sglang/actions/runs/31152370081)
- **分支**: `mick/masked-guard-sp-scope`
- **总耗时**: 31.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31152370081

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 10.2min | 其他 | 作业未执行实际测试，仅上传空失败目录后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152370081/job/92784585779) |

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未发现测试运行记录，且diffusion-failures目录无文件，可能因前置条件未满足或测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152370081/job/92784585779

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31152370081/job/92784585772) |


## [Run #31152070457](https://github.com/sgl-project/sglang/actions/runs/31152070457)
- **分支**: `codex/mm-qwen-owner-transport`
- **总耗时**: 8.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31152070457

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 3.9min | 环境问题 | 自定义容器执行失败，导致测试未运行即中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685780) |
| stage-b-test-4-npu-a3 (1) | 6.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685811) |
| stage-b-test-1-npu-a3 | 3.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685826) |
| multimodal-gen-test-2-npu-a3 | 5.3min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685829) |
| stage-b-test-8-npu-a3 | 6.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685846) |
| stage-b-test-4-npu-a3 (0) | 6.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685849) |
| stage-b-test-16-npu-a3 | 6.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685872) |
| multimodal-gen-test-1-npu-a3 | 3.9min | 环境问题 | 作业因环境问题失败，未找到失败产物文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685953) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.1min | 环境问题 | 作业在启动阶段即被终止，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783686094) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 6.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783686138) |

- **stage-b-test-2-npu-a3**: 作业在启用6个测试后，执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于基础设施/容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685780

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理或配置问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685811

- **stage-b-test-1-npu-a3**: 作业在运行NPU采样后端测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685826

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，随后清理退出。未看到实际测试执行或失败原因，可能为作业被跳过或提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685829

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是上游产物未正确上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685846

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685849

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储路径和生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685872

- **multimodal-gen-test-1-npu-a3**: 日志显示作业执行了上传产物步骤，但未找到diffusion-failures/目录下的文件，提示无产物上传。作业可能因测试未生成失败记录而提前结束，或测试本身未运行成功，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685953

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在准备阶段后立即进入清理流程，未运行测试用例，可能因runner环境异常或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783686094

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程资源（如模型权重或数据集）已被删除或路径错误，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783686138

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31152070457/job/92783685792) |


## [Run #31151481704](https://github.com/sgl-project/sglang/actions/runs/31151481704)
- **分支**: `online-nvfp4-to-mxfp4-convert`
- **总耗时**: 132.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31151481704

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 131.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976246) |
| stage-b-test-8-npu-a3 | 131.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取必要数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976251) |
| multimodal-gen-test-2-npu-a3 | 21.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976320) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 131.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976625) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976246

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志或工件时，Azure Blob 返回 BlobNotFound 错误，说明对应 blob 已被删除或路径错误，属于外部存储环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976251

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤，无法判断失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976320

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是资源被清理或上传失败，需检查数据源配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976625

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976268) |
| stage-b-test-2-npu-a3 | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976273) |
| stage-b-test-1-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976278) |
| stage-b-test-4-npu-a3 (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976287) |
| stage-b-test-4-npu-a3 (1) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976293) |
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976379) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31151481704/job/92781976694) |


## [Run #31150586167](https://github.com/sgl-project/sglang/actions/runs/31150586167)
- **分支**: `add-inkling-cache-consistency-test`
- **总耗时**: 105.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31150586167

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 11.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280915) |
| stage-b-test-4-npu-a3 (0) | 8.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280961) |
| stage-b-test-16-npu-a3 | 104.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280964) |
| stage-b-test-4-npu-a3 (1) | 8.1min | 环境问题 | 自定义容器执行失败，runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779281032) |
| stage-b-test-8-npu-a3 | 104.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779281050) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 104.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779281565) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅有GitHub Actions环境准备、Node版本警告及上传artifact（无文件）等常规步骤，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280915

- **stage-b-test-4-npu-a3 (0)**: 日志显示容器在加载模型权重时因导入错误（MultimodalDataItem缺失）而失败，随后自定义容器实现执行失败，作业被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280961

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280964

- **stage-b-test-4-npu-a3 (1)**: 测试在运行第二个用例test_npu_tp4_bf16.py时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于runner环境问题而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779281032

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在下载日志时遇到 BlobNotFound 错误，可能是日志文件被清理、路径错误或上传失败，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779281050

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传、被删除或配置的 URL 有误，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779281565

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a3 | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280919) |
| stage-b-test-1-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280925) |
| multimodal-gen-test-1-npu-a3 | 28.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779280951) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779281008) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31150586167/job/92779281655) |


## [Run #31148871900](https://github.com/sgl-project/sglang/actions/runs/31148871900)
- **分支**: `lsyin/ci-trim-h100`
- **总耗时**: 82.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31148871900

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 81.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229852) |
| stage-b-test-16-npu-a3 | 81.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229865) |
| stage-b-test-4-npu-a3 (0) | 81.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229886) |
| stage-b-test-4-npu-a3 (1) | 81.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229926) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 81.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774230149) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229852

- **stage-b-test-16-npu-a3**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置问题，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229865

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于基础设施/环境配置问题，需检查存储路径或资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229886

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229926

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储资源缺失或路径错误，可能是文件被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774230149

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229868) |
| stage-b-test-2-npu-a3 | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229871) |
| stage-b-test-1-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774229879) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148871900/job/92774230163) |


## [Run #31148827876](https://github.com/sgl-project/sglang/actions/runs/31148827876)
- **分支**: `main`
- **总耗时**: 9.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31148827876

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088787) |
| multimodal-gen-test-2-npu-a3 | 9.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088883) |
| base-b-test-4-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088887) |
| base-b-test-4-npu-a3 / run (1) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088927) |
| base-b-test-2-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088949) |
| base-b-test-1-npu-a3 / run (0) | 9.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088963) |
| base-b-test-16-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088974) |
| base-b-test-8-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088996) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774089139) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774089230) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774089272) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774089294) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088787

- **multimodal-gen-test-2-npu-a3**: 作业在下载依赖或数据时，请求的blob未找到（BlobNotFound），可能是文件被删除、路径错误或上传未完成，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088883

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088887

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088927

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088949

- **base-b-test-1-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088963

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088974

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的工件/缓存文件已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088996

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774089139

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774089230

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774089272

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774089294

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148827876/job/92774088873) |


## [Run #31148658092](https://github.com/sgl-project/sglang/actions/runs/31148658092)
- **分支**: `add-inkling-cache-consistency-test`
- **总耗时**: 36.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31148658092

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627666) |
| multimodal-gen-test-1-npu-a3 | 31.1min | 其他 | 作业日志被截断，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627680) |
| stage-b-test-8-npu-a3 | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627698) |
| stage-b-test-4-npu-a3 (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627701) |
| multimodal-gen-test-2-npu-a3 | 30.8min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传产物提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627716) |
| stage-b-test-4-npu-a3 (1) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627718) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627932) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627666

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示上传artifact时未找到diffusion-failures目录，说明测试可能未产生失败文件，但无法确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627680

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627698

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627701

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败或错误信息，仅有Node.js版本弃用警告及无失败产物上传提示，可能为作业被取消或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627716

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627718

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源（可能是模型权重或数据集）已被删除或路径错误，导致作业启动失败。需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627932

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a3 | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627687) |
| stage-b-test-1-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627714) |
| stage-a-unit-test-npu | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627743) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148658092/job/92773627899) |


## [Run #31148646246](https://github.com/sgl-project/sglang/actions/runs/31148646246)
- **分支**: `main`
- **总耗时**: 66.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31148646246

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (0) | 64.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760667) |
| stage-b-test-16-npu-a3 | 64.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760674) |
| stage-b-test-4-npu-a3 (1) | 64.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760687) |
| stage-b-test-1-npu-a3 | 27.3min | 代码错误 | HiCache MHA测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760715) |
| stage-b-test-8-npu-a3 | 64.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760732) |
| multimodal-gen-test-2-npu-a3 | 26.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760736) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 64.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760861) |

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760667

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760674

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760687

- **stage-b-test-1-npu-a3**: test_npu_hicache_mha.py测试执行失败（exit code 1），其余6个测试均通过，表明该测试用例存在代码或环境相关问题，需进一步查看具体报错信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760715

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760732

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未出现测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760736

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760861

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760684) |
| stage-b-test-2-npu-a3 | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760723) |
| multimodal-gen-test-1-npu-a3 | 35.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760724) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148646246/job/92773760873) |


## [Run #31148341542](https://github.com/sgl-project/sglang/actions/runs/31148341542)
- **分支**: `mick/ring-admission`
- **总耗时**: 27.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31148341542

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148341542/job/92772584251) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，只有GitHub Actions的常规准备、上传artifact（无文件）和清理步骤，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148341542/job/92772584251

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31148341542/job/92772584227) |


## [Run #31148118637](https://github.com/sgl-project/sglang/actions/runs/31148118637)
- **分支**: `main`
- **总耗时**: 11.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31148118637

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 9.4min | 环境问题 | 作业因缺少失败产物文件而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148118637/job/92771969768) |
| multimodal-gen-test-1-npu-a3 | 10.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31148118637/job/92771969828) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures产物时提示无文件，说明测试未产生失败样本，作业可能因环境或前置条件未满足而异常终止，未进入正常测试流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148118637/job/92771969768

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件），未包含multimodal测试执行过程或错误输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31148118637/job/92771969828


## [Run #31147784836](https://github.com/sgl-project/sglang/actions/runs/31147784836)
- **分支**: `add-inkling-cache-consistency-test`
- **总耗时**: 17.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31147784836

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 14.2min | 环境问题 | 自定义容器执行失败，模型加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895609) |
| multimodal-gen-test-1-npu-a3 | 16.2min | 环境问题 | 作业因缺少失败产物文件而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895610) |
| stage-b-test-2-npu-a3 | 14.2min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895618) |
| stage-b-test-16-npu-a3 | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895624) |
| stage-b-test-4-npu-a3 (1) | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895629) |
| stage-b-test-4-npu-a3 (0) | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895656) |
| stage-b-test-8-npu-a3 | 16.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895660) |
| multimodal-gen-test-2-npu-a3 | 16.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895663) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 16.4min | 环境问题 | Azure Blob 存储中指定的模型文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770896043) |

- **stage-b-test-1-npu-a3**: 作业在加载模型权重时，自定义容器实现执行失败，导致进程终止。日志显示模型加载到0%时出现错误，可能是容器环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895609

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤未找到diffusion-failures/目录下的文件，说明测试未产生失败样本，作业可能因前置条件未满足或测试未运行而终止，属于环境或流程配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895610

- **stage-b-test-2-npu-a3**: 作业在加载模型权重分片时（75%进度）容器执行失败，错误提示需联系自托管runner管理员，属于NPU环境或容器基础设施问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895618

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895624

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895629

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895656

- **stage-b-test-8-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895660

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895663

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明模型权重或数据文件在 Azure Blob 中缺失或路径错误，可能是上传失败或配置错误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770896043

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770895674) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31147784836/job/92770896016) |


---
*Auto-generated by npu_pr_monitor.py*