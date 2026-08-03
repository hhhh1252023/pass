# NPU CI 执行监控
**生成时间**: 2026-08-03 00:25 UTC
**分析 Run 数**: 26

---

## [Run #30770897825](https://github.com/sgl-project/sglang/actions/runs/30770897825)
- **分支**: `kimi-k3`
- **总耗时**: 47.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30770897825

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789694) |
| stage-b-test-8-npu-a3 | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789707) |
| stage-b-test-16-npu-a3 | 46.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789708) |
| stage-b-test-4-npu-a3 (0) | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789709) |
| stage-b-test-4-npu-a3 (1) | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789718) |
| multimodal-gen-test-2-npu-a3 | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789725) |
| stage-b-test-1-npu-a3 | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789746) |
| multimodal-gen-test-1-npu-a3 | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789760) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789931) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789945) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789963) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789964) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 46.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789974) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789976) |

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789694

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789707

- **stage-b-test-16-npu-a3**: 作业 stage-b-test-16-npu-a3 在尝试下载或访问 Azure Blob 中的日志文件时，返回 BlobNotFound 错误（HTTP 404）。这通常是因为日志文件已被删除、路径错误或上传未完成，属于外部存储环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789708

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789709

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的构建产物或数据文件未上传或已被删除，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789718

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789725

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789746

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或数据文件未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789760

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789931

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的模型或数据文件在 Azure Blob 中缺失，可能是文件被误删或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789945

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789963

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或测试数据未上传到指定存储路径，或路径配置错误。建议检查存储账户、容器名及 blob 路径是否正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789964

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被删除、路径错误或上传未完成。这属于外部依赖环境问题，需检查CI配置中的blob路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789974

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程存储对象缺失或路径错误，可能是资源未上传或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789976

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789703) |
| stage-b-test-2-npu-a2 (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789726) |
| stage-b-test-2-npu-a2 (1) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789730) |
| stage-b-test-1-npu-a2 (0) | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789740) |
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770897825/job/91557789774) |


## [Run #30770843961](https://github.com/sgl-project/sglang/actions/runs/30770843961)
- **分支**: `main`
- **总耗时**: 53.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30770843961

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670001) |
| stage-b-test-4-npu-a3 (1) | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670007) |
| multimodal-gen-test-2-npu-a3 | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670008) |
| stage-b-test-16-npu-a3 | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670012) |
| multimodal-gen-test-1-npu-a3 | 53.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670013) |
| stage-b-test-8-npu-a3 | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670017) |
| stage-b-test-4-npu-a3 (0) | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670019) |
| stage-b-test-2-npu-a3 | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670060) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 53.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670205) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670247) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 53.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670251) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670269) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 53.1min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670270) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 53.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670273) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670001

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670007

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670008

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670012

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志或依赖文件未上传或路径错误，属于环境配置或资源缺失问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670013

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670017

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物未上传或路径错误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670019

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670060

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明CI作业尝试访问的远程存储对象缺失，可能是构建产物或依赖文件未正确上传，或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670205

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是配置问题或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670247

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程存储对象缺失，可能是文件被删除、路径错误或上传失败，属于环境或资源准备问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670251

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670269

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI作业依赖的远程日志文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670270

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明作业依赖的 Azure 存储资源缺失或路径错误，可能是数据未上传、被删除或配置有误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670273

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 35.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557669996) |
| stage-b-test-2-npu-a2 (0) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670002) |
| stage-b-test-2-npu-a2 (1) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670016) |
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670021) |
| stage-b-test-1-npu-a2 (1) | 32.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770843961/job/91557670029) |


## [Run #30770723230](https://github.com/sgl-project/sglang/actions/runs/30770723230)
- **分支**: `cheng/gc-wb-stack-review`
- **总耗时**: 12.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30770723230

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865155) |
| multimodal-gen-test-2-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865160) |
| multimodal-gen-test-1-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865163) |
| stage-b-test-2-npu-a2 (0) | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865169) |
| stage-b-test-2-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865173) |
| stage-b-test-8-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865177) |
| stage-b-test-4-npu-a3 (1) | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865188) |
| stage-b-test-2-npu-a2 (1) | 2.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865193) |
| stage-b-test-16-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865194) |
| stage-b-test-4-npu-a3 (0) | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865195) |
| stage-b-test-1-npu-a2 (0) | 5.3min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865233) |
| stage-b-test-1-npu-a2 (1) | 2.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865246) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865415) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 11.6min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865419) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 11.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865420) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865436) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865437) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865442) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、文件被清理或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865155

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865160

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865163

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的 blob 路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865169

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865173

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据文件在 Azure Blob 存储中已被删除或路径错误，可能是上游作业未成功上传或存储配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865177

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865188

- **stage-b-test-2-npu-a2 (1)**: 日志显示在安装依赖包（如psutil）后，执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865193

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865194

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865195

- **stage-b-test-1-npu-a2 (0)**: 日志显示作业在加载模型权重（Multi-thread loading shards 50%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865233

- **stage-b-test-1-npu-a2 (1)**: 日志显示在安装依赖后，执行自定义容器实现时失败，报错提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865246

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865415

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示BlobNotFound错误，说明CI作业依赖的某个文件或数据在Azure Blob存储中缺失，可能是上传失败、路径错误或文件被删除，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865419

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误信息为BlobNotFound，说明作业依赖的某个blob（可能是模型权重、测试数据或配置文件）在存储中缺失或路径错误，需检查相关资源是否存在或上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865420

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865436

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的某个文件（可能是模型权重或测试数据）不存在或已被删除，导致作业启动失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865437

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的 Azure Blob 存储资源缺失或路径错误，可能是数据未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865442

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30770723230/job/91557865165) |


## [Run #30768976021](https://github.com/sgl-project/sglang/actions/runs/30768976021)
- **分支**: `main`
- **总耗时**: 50.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30768976021

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732344) |
| stage-b-test-2-npu-a3 | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732361) |
| stage-b-test-1-npu-a3 | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732365) |
| stage-b-test-4-npu-a3 (1) | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732384) |
| stage-b-test-4-npu-a3 (0) | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732397) |
| stage-b-test-16-npu-a3 | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732401) |
| multimodal-gen-test-2-npu-a3 | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732403) |
| multimodal-gen-test-1-npu-a3 | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732415) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732750) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 49.8min | 环境问题 | 日志显示Azure Blob存储返回BlobNotFound错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732770) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732771) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732784) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732796) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 49.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732800) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732344

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732361

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）未上传或已被删除，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732365

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732384

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732397

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732401

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732403

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732415

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732750

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 作业在访问Azure Blob存储时，指定的blob不存在（BlobNotFound），可能是依赖的模型权重或数据文件未上传或路径错误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732770

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732771

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732784

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732796

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732800

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732346) |
| stage-b-test-2-npu-a2 (1) | 10.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732359) |
| stage-b-test-2-npu-a2 (0) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732379) |
| stage-b-test-1-npu-a2 (0) | 35.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732404) |
| stage-b-test-1-npu-a2 (1) | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30768976021/job/91552732407) |


## [Run #30765936935](https://github.com/sgl-project/sglang/actions/runs/30765936935)
- **分支**: `cheng/gc-fix-unpublished-config-tests`
- **总耗时**: 131.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30765936935

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601596) |
| stage-b-test-1-npu-a3 | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601604) |
| multimodal-gen-test-2-npu-a3 | 130.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601605) |
| stage-b-test-4-npu-a3 (0) | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601612) |
| stage-b-test-4-npu-a3 (1) | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601616) |
| stage-b-test-2-npu-a3 | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601617) |
| stage-b-test-8-npu-a3 | 130.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601629) |
| stage-b-test-16-npu-a3 | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601631) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601808) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601838) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 130.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601840) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601844) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601856) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 130.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601873) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601596

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601604

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业依赖的某个blob文件缺失或路径错误，可能是上传失败、文件被删除或配置的URL有误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601605

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境或依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601612

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601616

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601617

- **stage-b-test-8-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储中缺失，可能是上游构建未成功上传或路径配置错误，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601629

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是测试数据或构建产物缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601631

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601808

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查 blob 路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601838

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中不存在，可能是文件被删除、路径错误或上传未完成。这属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601840

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601844

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件（如模型权重或测试数据）在 Azure Blob 存储中缺失或路径错误，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601856

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601873

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601597) |
| stage-b-test-1-npu-a2 (1) | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601598) |
| stage-b-test-1-npu-a2 (0) | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601613) |
| stage-b-test-2-npu-a2 (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601623) |
| stage-b-test-2-npu-a2 (1) | 11.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30765936935/job/91544601628) |


## [Run #30761679497](https://github.com/sgl-project/sglang/actions/runs/30761679497)
- **分支**: `dev/dlal/norm-quant-fusion`
- **总耗时**: 307.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30761679497

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (0) | 38.5min | 代码错误 | NPU DP注意力测试失败，测试脚本返回非零退出码。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329888) |
| multimodal-gen-test-2-npu-a3 | 34.0min | 其他 | 作业上传了失败产物但未显示明确失败原因，可能为测试失败或环境问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329918) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 78.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91555982564) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 77.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91556131792) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 74.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91556378937) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 74.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91556389000) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 72.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91556646228) |
| single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms | 45.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91559122306) |

- **stage-b-test-4-npu-a3 (0)**: test_npu_dp_attention.py 测试失败（exit code 1），耗时1622秒远超预估400秒，可能因代码逻辑错误或环境问题导致断言失败，需检查该测试具体报错信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329888

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-1.zip产物，但未提供具体测试失败信息，仅见Node.js弃用警告，需查看产物内容确认失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329918

- **single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是日志清理或配置问题，需检查存储路径或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91555982564

- **single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16**: 日志显示 BlobNotFound 错误，说明作业依赖的模型权重或数据文件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置错误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91556131792

- **single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91556378937

- **single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的 blob 资源缺失或路径错误，可能是日志上传失败、过期清理或配置问题，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91556389000

- **single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储对象缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91556646228

- **single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91559122306

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 12.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329801) |
| stage-b-test-1-npu-a3 | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329816) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329817) |
| stage-b-test-1-npu-a2 (0) | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329820) |
| stage-b-test-2-npu-a3 | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329852) |
| stage-b-test-4-npu-a3 (1) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329868) |
| stage-b-test-16-npu-a3 | 71.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329870) |
| stage-b-test-2-npu-a2 (1) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329874) |
| stage-b-test-2-npu-a2 (0) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329876) |
| stage-b-test-1-npu-a2 (1) | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329883) |
| multimodal-gen-test-1-npu-a3 | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533329919) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533330156) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 48.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533330160) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533330167) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533330184) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533330185) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761679497/job/91533330201) |


## [Run #30761411334](https://github.com/sgl-project/sglang/actions/runs/30761411334)
- **分支**: `cheng/gc-wb-stack-review`
- **总耗时**: 249.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30761411334

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (0) | 38.0min | 代码错误 | NPU DP注意力测试失败，测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625868) |
| stage-b-test-16-npu-a3 | 57.4min | 环境问题 | 自定义容器执行失败，runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625871) |
| stage-b-test-2-npu-a3 | 20.0min | 代码错误 | NPU MoE dense TP size 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625873) |
| multimodal-gen-test-2-npu-a3 | 37.8min | 精度回归 | 多模态生成测试失败，上传了diffusion-failures工件，表明输出与预期不符。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625970) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 21.9min | 其他 | 日志被截断，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532626355) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 41.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91553258543) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 40.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91553330171) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 36.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91553695474) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 34.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91553953859) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91554263343) |

- **stage-b-test-4-npu-a3 (0)**: test_npu_dp_attention.py测试失败，退出码1，耗时1629秒远超预估400秒，可能涉及DP注意力功能缺陷或环境配置问题，需查看具体断言错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625868

- **stage-b-test-16-npu-a3**: 测试运行到第4/5个用例时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于runner环境或容器配置问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625871

- **stage-b-test-2-npu-a3**: test_npu_moe_dense_tp_size.py 测试用例执行失败（exit code 1），其余2个相关测试通过，表明该测试存在代码或配置问题，需检查该测试的具体断言或环境配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625873

- **multimodal-gen-test-2-npu-a3**: 作业运行约37分钟后上传了diffusion-failures-npu-2-1.zip工件，包含7个文件，说明多模态生成测试存在精度偏差或输出错误，导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625970

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 提供的日志仅包含作业启动和清理阶段，未显示测试执行或失败的具体错误信息，无法判断是性能、精度还是环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532626355

- **single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或数据文件未上传或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91553258543

- **single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16**: 作业在尝试下载或访问某个blob资源时，返回BlobNotFound错误，可能是模型权重或数据文件未上传或路径错误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91553330171

- **single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明作业依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查 CI 配置中的 blob 引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91553695474

- **single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91553953859

- **single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91554263343

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 (1) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625859) |
| stage-b-test-1-npu-a3 | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625869) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625874) |
| stage-b-test-1-npu-a2 (0) | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625875) |
| stage-b-test-2-npu-a2 (1) | 11.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625916) |
| stage-b-test-2-npu-a2 (0) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625917) |
| stage-b-test-1-npu-a2 (1) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625918) |
| multimodal-gen-test-1-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625924) |
| stage-b-test-8-npu-a3 | 13.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532625937) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532626284) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532626334) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532626360) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532626367) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 52.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30761411334/job/91532626369) |


## [Run #30760972101](https://github.com/sgl-project/sglang/actions/runs/30760972101)
- **分支**: `cheng/gc-wb-stack-review`
- **总耗时**: 11.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30760972101

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 11.0min | 环境问题 | 自定义容器执行失败，可能是NPU资源或容器环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313556) |
| stage-b-test-4-npu-a3 (0) | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313566) |
| stage-b-test-2-npu-a2 (1) | 8.6min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器问题导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313569) |
| stage-b-test-4-npu-a3 (1) | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313577) |
| stage-b-test-1-npu-a3 | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313583) |
| stage-b-test-2-npu-a3 | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313586) |
| stage-b-test-1-npu-a2 (0) | 8.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313595) |
| stage-b-test-16-npu-a3 | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313603) |
| stage-b-test-1-npu-a2 (1) | 10.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313615) |
| multimodal-gen-test-2-npu-a3 | 11.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313621) |
| multimodal-gen-test-1-npu-a3 | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313625) |
| stage-b-test-8-npu-a3 | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313634) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 11.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313952) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313981) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313982) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313985) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 11.2min | 环境问题 | 日志显示Azure Blob存储中找不到指定文件，属于外部依赖资源缺失。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531314022) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 11.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531314024) |

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行到52%时，自定义容器实现执行失败，提示联系自托管runner管理员。此前无报错，可能为NPU设备故障、容器被终止或资源限制。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313556

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313566

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313569

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313577

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成产物，或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313583

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313586

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313595

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明作业依赖的 Azure Blob 存储资源缺失或路径错误，可能是资源被清理、配置错误或网络问题，需检查存储配置和资源可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313603

- **stage-b-test-1-npu-a2 (1)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313615

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业依赖的存储对象缺失，可能是文件被删除、路径错误或上传失败，属于基础设施或配置问题，需检查CI脚本中的blob引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313621

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313625

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313634

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 错误码BlobNotFound表明CI系统尝试访问的远程存储文件缺失或路径错误，可能是资源未上传或配置有误，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313952

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313981

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313982

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313985

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业在获取测试所需数据或模型文件时，Azure Blob返回BlobNotFound错误，说明文件不存在或路径错误，导致作业无法启动，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531314022

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531314024

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760972101/job/91531313604) |


## [Run #30760320046](https://github.com/sgl-project/sglang/actions/runs/30760320046)
- **分支**: `main`
- **总耗时**: 231.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30760320046

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 29.6min | 其他 | 作业上传了失败工件，但日志未显示具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623884) |
| stage-b-test-4-npu-a3 (0) | 69.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623885) |
| stage-b-test-16-npu-a3 | 61.5min | 环境问题 | 自定义容器执行失败，NPU图捕获过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623891) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 38.8min | 精度回归 | qwen3_vl_8b_thinking_1p_mmmu 精度测试失败，未生成 metrics.json 指标文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529624198) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 14.8min | 其他 | 日志被截断，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529624220) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 25.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550174508) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 24.4min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550302752) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 22.3min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550514575) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550701727) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 17.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550999796) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-1.zip工件，表明测试有失败，但未提供具体错误信息，可能是测试断言失败或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623884

- **stage-b-test-4-npu-a3 (0)**: 作业在模型加载和TP初始化阶段，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于NPU自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623885

- **stage-b-test-16-npu-a3**: 作业在NPU图捕获阶段（bs=48）时，自定义容器实现执行失败，导致作业终止。可能是容器环境或NPU资源问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623891

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 作业在运行精度测试后未找到 /tmp/metrics.json，说明测试未正常完成或结果未保存，可能因模型输出精度不达标或测试脚本异常导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529624198

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 提供的日志仅包含作业启动和清理阶段，未显示测试执行或失败的具体错误信息，无法判断是性能、精度还是环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529624220

- **single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或模型权重在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550174508

- **single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16**: 作业尝试下载或访问一个不存在的Blob资源（BlobNotFound），可能是日志上传失败、路径错误或资源被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550302752

- **single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI作业尝试下载的日志或数据文件在Azure Blob存储中已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550514575

- **single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550701727

- **single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91550999796

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a3 | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623831) |
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623832) |
| stage-b-test-4-npu-a3 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623837) |
| stage-b-test-2-npu-a3 | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623842) |
| stage-b-test-1-npu-a2 (1) | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623847) |
| stage-b-test-1-npu-a2 (0) | 37.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623850) |
| stage-b-test-2-npu-a2 (1) | 10.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623851) |
| stage-b-test-2-npu-a2 (0) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623863) |
| multimodal-gen-test-1-npu-a3 | 36.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623868) |
| stage-b-test-8-npu-a3 | 12.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529623907) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529624152) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529624177) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529624200) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30760320046/job/91529624215) |


## [Run #30759947570](https://github.com/sgl-project/sglang/actions/runs/30759947570)
- **分支**: `codex/minimax-h3-2gpu-consistency`
- **总耗时**: 202.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30759947570

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 30.5min | 其他 | 作业上传了失败产物，但日志未显示具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30759947570/job/91528603287) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-1.zip，表明测试有失败，但未提供具体错误信息，可能是测试断言失败或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30759947570/job/91528603287

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30759947570/job/91528603284) |


## [Run #30759413084](https://github.com/sgl-project/sglang/actions/runs/30759413084)
- **分支**: `codex/minimax-h3-2gpu-consistency`
- **总耗时**: 14.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30759413084

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30759413084/job/91527200945) |
| multimodal-gen-test-2-npu-a3 | 14.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30759413084/job/91527200950) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储对象已被删除或路径错误，可能是CI配置中引用的模型权重或数据文件缺失，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30759413084/job/91527200945

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是构建产物或依赖文件未正确上传，或路径配置错误。建议检查CI流程中blob的上传与引用逻辑。
  链接: https://github.com/sgl-project/sglang/actions/runs/30759413084/job/91527200950


## [Run #30758049670](https://github.com/sgl-project/sglang/actions/runs/30758049670)
- **分支**: `kan/rust-server-pd-restack`
- **总耗时**: 254.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30758049670

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 31.8min | 其他 | 作业上传了失败产物，但日志未显示具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637970) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 74.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91541454539) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 63.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91542506799) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 61.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91542669657) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 61.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91542709500) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 57.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91543162553) |
| single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms | 50.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91543836309) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-1.zip产物，表明测试有失败项，但未提供具体错误信息，需查看该产物或完整日志定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637970

- **single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是临时文件被清理或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91541454539

- **single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败或路径错误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91542506799

- **single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91542669657

- **single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms**: 作业日志返回BlobNotFound错误，表明CI系统尝试访问的存储对象缺失，可能是日志上传或下载路径配置错误，或存储被清理，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91542709500

- **single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91543162553

- **single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91543836309

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 12.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637922) |
| stage-b-test-2-npu-a3 | 19.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637923) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637928) |
| stage-b-test-1-npu-a3 | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637930) |
| stage-b-test-1-npu-a2 (0) | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637957) |
| stage-b-test-16-npu-a3 | 76.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637958) |
| stage-b-test-1-npu-a2 (1) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637964) |
| stage-b-test-4-npu-a3 (1) | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637967) |
| stage-b-test-2-npu-a2 (1) | 11.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637973) |
| multimodal-gen-test-1-npu-a3 | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637975) |
| stage-b-test-2-npu-a2 (0) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637976) |
| stage-b-test-4-npu-a3 (0) | 86.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523637981) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523638236) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523638247) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 52.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523638249) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523638258) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523638268) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30758049670/job/91523638275) |


## [Run #30757160235](https://github.com/sgl-project/sglang/actions/runs/30757160235)
- **分支**: `codex/minimax-h3-2gpu-consistency`
- **总耗时**: 60.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30757160235

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 60.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30757160235/job/91521240216) |
| multimodal-gen-test-2-npu-a3 | 60.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30757160235/job/91521240265) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30757160235/job/91521240216

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30757160235/job/91521240265


## [Run #30756042886](https://github.com/sgl-project/sglang/actions/runs/30756042886)
- **分支**: `codex/minimax-h3-2gpu-consistency`
- **总耗时**: 29.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30756042886

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 29.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30756042886/job/91518314850) |
| multimodal-gen-test-1-npu-a3 | 29.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30756042886/job/91518314854) |

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业依赖的某个blob资源已被删除或路径错误，可能是CI配置中引用的工件或数据未正确上传，需检查相关存储路径及上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30756042886/job/91518314850

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30756042886/job/91518314854


## [Run #30754702145](https://github.com/sgl-project/sglang/actions/runs/30754702145)
- **分支**: `main`
- **总耗时**: 150.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30754702145

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 148.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854250) |
| stage-b-test-4-npu-a3 (1) | 148.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854259) |
| stage-b-test-1-npu-a3 | 2.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854261) |
| stage-b-test-4-npu-a3 (0) | 148.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854263) |
| stage-b-test-2-npu-a3 | 1.8min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854266) |
| multimodal-gen-test-1-npu-a3 | 2.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854292) |
| multimodal-gen-test-2-npu-a3 | 3.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854294) |
| stage-b-test-16-npu-a3 | 148.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取必要数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854310) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 8.1min | 其他 | 日志被截断，未显示测试执行结果，仅见清理和上传步骤，无法确定失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854469) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.4min | 其他 | 作业日志不完整，未显示实际失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854492) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 3.2min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854497) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 3.8min | 其他 | 作业日志被截断，未显示实际失败原因，仅见上传metrics.json失败及Node20弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854500) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 7.5min | 其他 | 日志被截断，未显示测试执行结果，仅见作业清理和Node.js弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854526) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91528560052) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854250

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854259

- **stage-b-test-1-npu-a3**: 在安装triton-ascend依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是容器环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854261

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854263

- **stage-b-test-2-npu-a3**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业中断，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854266

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854292

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传artifact时未找到diffusion-failures目录，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854294

- **stage-b-test-16-npu-a3**: 作业 stage-b-test-16-npu-a3 在尝试下载或访问 Azure Blob 中的某个 blob 时，返回 BlobNotFound 错误（HTTP 404）。这通常是因为日志或工件已被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854310

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志中间部分被省略，仅显示作业开始、输入参数和结束时的清理步骤，未包含测试运行输出或错误信息，因此无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854469

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志在测试执行前中断，未包含测试运行结果或错误信息，仅有runner初始化、依赖下载和清理步骤，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854492

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志仅包含作业初始化和清理步骤，中间部分被省略，未出现测试失败、错误或超时信息，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854497

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志中间部分省略，无法定位具体错误。可见信息仅为未找到/tmp/metrics.json导致无产物上传，以及Node.js 20弃用警告，均非直接失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854500

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志中间部分省略，无法定位具体失败原因。仅看到作业结束时的清理步骤和Node.js 20弃用警告，未出现测试失败或错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854526

- **single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91528560052

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854251) |
| stage-b-test-2-npu-a2 (1) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854265) |
| stage-b-test-1-npu-a2 (0) | 37.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854289) |
| stage-b-test-1-npu-a2 (1) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854295) |
| stage-b-test-2-npu-a2 (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854307) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30754702145/job/91514854482) |


## [Run #30753554887](https://github.com/sgl-project/sglang/actions/runs/30753554887)
- **分支**: `agent/update-diffusion-skills-h3`
- **总耗时**: 198.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30753554887

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 41.1min | 其他 | 作业上传了diffusion-failures工件，表明测试存在失败用例，但日志未显示具体错误原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753554887/job/91516476446) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传了diffusion-failures-npu-2-2.zip工件，说明multimodal生成测试中有失败案例，但未提供具体失败详情，需下载工件进一步分析。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753554887/job/91516476446

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30753554887/job/91516476439) |


## [Run #30753351449](https://github.com/sgl-project/sglang/actions/runs/30753351449)
- **分支**: `main`
- **总耗时**: 35.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30753351449

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243966) |
| stage-b-test-1-npu-a2 (0) | 34.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243968) |
| stage-b-test-2-npu-a3 | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243972) |
| stage-b-test-16-npu-a3 | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243977) |
| stage-b-test-4-npu-a3 (0) | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243983) |
| multimodal-gen-test-1-npu-a3 | 35.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243987) |
| stage-b-test-8-npu-a3 | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243996) |
| multimodal-gen-test-2-npu-a3 | 35.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243999) |
| stage-b-test-4-npu-a3 (1) | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244020) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 35.0min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244236) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244260) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244284) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244305) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 35.0min | 环境问题 | Azure Blob存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244309) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244327) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243966

- **stage-b-test-1-npu-a2 (0)**: 作业在测试运行约4分钟后，日志显示"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243968

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243972

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243977

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243983

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传，或路径配置错误。建议检查CI作业中引用的blob路径及上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243987

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243996

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的依赖文件或工件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243999

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244020

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示BlobNotFound错误，说明CI系统尝试下载的日志或工件在Azure Blob存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244236

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程存储资源缺失或路径错误，可能是配置问题或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244260

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除或路径错误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244284

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，请求的资源在存储中不存在。可能是 CI 配置引用了已删除或未上传的 blob，或路径错误。需检查作业依赖的 blob 是否已正确上传，或更新配置指向有效资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244305

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI作业依赖的某个文件（如模型权重或测试数据）在Azure Blob存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244309

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是模型权重或数据文件未上传到指定存储路径，或路径配置错误。需检查 CI 作业中引用的 blob 名称和存储账户是否正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511244327

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243955) |
| stage-b-test-2-npu-a2 (0) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243965) |
| stage-b-test-2-npu-a2 (1) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243975) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30753351449/job/91511243976) |


## [Run #30753271862](https://github.com/sgl-project/sglang/actions/runs/30753271862)
- **分支**: `codex/minimax-h3-2gpu-consistency`
- **总耗时**: 73.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30753271862

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 72.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753271862/job/91511009099) |
| multimodal-gen-test-2-npu-a3 | 72.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30753271862/job/91511009105) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753271862/job/91511009099

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业依赖的某个文件或数据在存储中缺失，可能是资源未上传、路径错误或已被删除，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30753271862/job/91511009105


## [Run #30752785051](https://github.com/sgl-project/sglang/actions/runs/30752785051)
- **分支**: `rainj-me/rust-server-dp-attn-client-lb`
- **总耗时**: 403.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30752785051

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 37.5min | 精度回归 | 多模态生成测试失败，上传了diffusion-failures工件，表明输出与预期不符。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114642) |
| single-node-poc (qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu... / qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms | 108.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91540486140) |
| single-node-poc (glm4_7_flash_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/glm4_... / glm4_7_flash_1p_gsm8k | 102.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91541026652) |
| single-node-poc (qwen3_vl_30b_a3b_bf16_2p_gsm8k, linux-aarch64-a3-4, test/registered/ascend/accur... / qwen3_vl_30b_a3b_bf16_2p_gsm8k | 86.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91542593852) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16, test/registered/ascend/acc... / glm5_top64_pruned_bf16_8p_gsm8k | 80.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91543213311) |
| single-node-poc (moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/... / moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k | 77.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91543553883) |
| single-node-poc (qwen3_5_9b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/qw... / qwen3_5_9b_bf16_1p_gsm8k | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91548682864) |

- **multimodal-gen-test-2-npu-a3**: 作业上传了diffusion-failures-npu-2-2.zip工件，说明多模态生成测试存在diffusion结果失败，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114642

- **single-node-poc (qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu... / qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明作业依赖的远程存储对象缺失或路径错误，可能是文件被删除、未上传或配置有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91540486140

- **single-node-poc (glm4_7_flash_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/glm4_... / glm4_7_flash_1p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或数据在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91541026652

- **single-node-poc (qwen3_vl_30b_a3b_bf16_2p_gsm8k, linux-aarch64-a3-4, test/registered/ascend/accur... / qwen3_vl_30b_a3b_bf16_2p_gsm8k**: 日志显示 BlobNotFound 错误，请求的资源在存储中不存在，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91542593852

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16, test/registered/ascend/acc... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明作业依赖的远程存储对象缺失或路径错误，可能是文件被删除、上传失败或配置的 URL 有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91543213311

- **single-node-poc (moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/... / moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或数据集）已被删除或路径错误，属于环境配置或资源缺失问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91543553883

- **single-node-poc (qwen3_5_9b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/qw... / qwen3_5_9b_bf16_1p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储资源缺失或路径错误，可能是配置问题或资源被清理，需检查相关 blob 路径及权限。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91548682864

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114637) |
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114638) |
| stage-b-test-1-npu-a2 (0) | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114639) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114643) |
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114644) |
| stage-b-test-1-npu-a2 (1) | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114645) |
| stage-b-test-2-npu-a2 (0) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114651) |
| stage-b-test-2-npu-a2 (1) | 11.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114657) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114927) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114933) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114957) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 46.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114976) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114989) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 13.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91511114997) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91524016780) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91524247763) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91524257441) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91524604292) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91524861485) |
| single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms | 27.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752785051/job/91528721839) |


## [Run #30752603413](https://github.com/sgl-project/sglang/actions/runs/30752603413)
- **分支**: `dev/dlal/norm-quant-fusion`
- **总耗时**: 241.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30752603413

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 34.7min | 其他 | 作业日志被截断，未显示实际失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267601) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.2min | 其他 | 日志被截断，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267758) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 8.1min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91519836079) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 132.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91520014739) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 116.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91521663342) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 114.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91521876777) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 110.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91522225374) |
| single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms | 87.2min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91524529006) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分省略，无法定位具体失败步骤。仅看到上传diffusion-failures目录时提示无文件，可能测试未生成失败产物或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267601

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 提供的日志仅包含作业启动和清理信息，未显示测试执行或失败的具体错误。可能因日志截断导致关键信息缺失，需查看完整日志以判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267758

- **single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms**: 日志中未包含测试运行的具体输出或错误信息，仅显示runner初始化、依赖下载和作业清理步骤，无法判断失败原因，可能为日志截断或作业在测试前已终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91519836079

- **single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败或配置变更，需检查相关 blob 是否存在及路径配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91520014739

- **single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或数据在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91521663342

- **single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91521876777

- **single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91522225374

- **single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms**: 日志显示BlobNotFound错误，说明CI系统尝试下载的日志文件已被删除或路径错误，属于基础设施问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91524529006

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267598) |
| multimodal-gen-test-1-npu-a3 | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267599) |
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267606) |
| stage-b-test-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267607) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267608) |
| stage-b-test-1-npu-a2 (0) | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267616) |
| stage-b-test-2-npu-a2 (1) | 10.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267617) |
| stage-b-test-2-npu-a2 (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267619) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267729) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267739) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267747) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 45.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267753) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 13.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752603413/job/91509267766) |


## [Run #30752336761](https://github.com/sgl-project/sglang/actions/runs/30752336761)
- **分支**: `main`
- **总耗时**: 6.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30752336761

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取必要数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500810) |
| multimodal-gen-test-1-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500825) |
| stage-b-test-2-npu-a2 (1) | 3.4min | 环境问题 | 自定义容器执行失败，构建xatlas依赖时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500832) |
| stage-b-test-2-npu-a2 (0) | 1.3min | 环境问题 | 自定义容器执行失败，导致作业在环境准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500834) |
| stage-b-test-1-npu-a2 (1) | 5.5min | 环境问题 | NPU容器在加载权重时崩溃，自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500835) |
| stage-b-test-1-npu-a2 (0) | 4.5min | 环境问题 | 自定义容器执行失败，测试启动后立即中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500844) |
| stage-b-test-4-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500852) |
| multimodal-gen-test-2-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500855) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501156) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501171) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 5.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501190) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501199) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501213) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 5.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501244) |

- **stage-b-test-16-npu-a3**: 作业 stage-b-test-16-npu-a3 在尝试下载日志或工件时，Azure Blob 返回 BlobNotFound 错误，说明对应 blob 已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500810

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500825

- **stage-b-test-2-npu-a2 (1)**: 作业在编译xatlas包时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500832

- **stage-b-test-2-npu-a2 (0)**: 日志显示在安装依赖后，执行自定义容器实现时失败，报错提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500834

- **stage-b-test-1-npu-a2 (1)**: 作业在Load weight阶段（avail mem=60.52 GB）后立即报错，提示自定义容器实现执行失败，可能是NPU设备或容器环境异常导致进程终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500835

- **stage-b-test-1-npu-a2 (0)**: 测试test_npu_hicache_mha.py刚开始执行（test_a_gsm8k），容器实现即报错失败，属于自托管runner环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500844

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500852

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境/依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500855

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传、被删除或配置的 URL 有误，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501156

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501171

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业在尝试访问Azure Blob存储时，因指定的blob不存在（BlobNotFound）而失败。这可能是由于日志文件或依赖资源未正确上传，或路径配置错误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501190

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501199

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到。可能是资源被误删、路径错误或上传失败，需检查 CI 配置中的 blob 路径及资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501213

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 错误码BlobNotFound表明CI作业尝试访问的远程资源（如模型权重或测试数据）在存储中缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508501244

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30752336761/job/91508500809) |


## [Run #30749647330](https://github.com/sgl-project/sglang/actions/runs/30749647330)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 9.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30749647330

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (1) | 3.7min | 环境问题 | 自定义容器执行失败，导致测试未运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306466) |
| stage-b-test-2-npu-a2 (0) | 4.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306468) |
| stage-b-test-1-npu-a2 (1) | 2.9min | 环境问题 | 下载依赖包时代理服务器返回500错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306473) |
| multimodal-gen-test-2-npu-a3 | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306474) |
| stage-b-test-16-npu-a3 | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306475) |
| stage-a-unit-test-npu | 3.0min | 环境问题 | 下载NPU依赖包时代理服务器返回500错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306477) |
| stage-b-test-4-npu-a3 | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306478) |
| multimodal-gen-test-1-npu-a3 | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306483) |
| stage-b-test-1-npu-a2 (0) | 2.6min | 环境问题 | 下载依赖包时代理服务器返回500错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306484) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306609) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306626) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 8.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306634) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306635) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306637) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306641) |

- **stage-b-test-2-npu-a2 (1)**: 作业在运行测试命令时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是自托管runner环境或容器配置问题，测试未实际执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306466

- **stage-b-test-2-npu-a2 (0)**: 作业在启动NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境问题而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306468

- **stage-b-test-1-npu-a2 (1)**: 在下载ops-transformer-2026.7.27-torch2.10.0-cann9.0.0-910b-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回500 INTERNAL SERVER ERROR，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306473

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306474

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败或配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306475

- **stage-a-unit-test-npu**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-910b-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回500 INTERNAL SERVER ERROR，导致依赖安装失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306477

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306478

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的预构建产物或数据文件未上传或已被删除，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306483

- **stage-b-test-1-npu-a2 (0)**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-910b-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回500 INTERNAL SERVER ERROR，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306484

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业日志中返回BlobNotFound错误，表明CI流程尝试访问的Azure Blob存储资源缺失或路径错误，可能是配置问题或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306609

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的模型权重或数据文件未上传到指定存储位置，可能是文件缺失或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306626

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的某个blob（可能是模型权重、测试数据或配置）已被删除或路径错误，需检查相关资源是否存在及路径配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306634

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306635

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）未上传或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306637

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749647330/job/91501306641


## [Run #30749612101](https://github.com/sgl-project/sglang/actions/runs/30749612101)
- **分支**: `fix-nemotron-nano-fp4-empty-output`
- **总耗时**: 611.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30749612101

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 2.5min | 环境问题 | 下载sgl-kernel-npu依赖时代理服务器返回500错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274164) |
| stage-b-test-1-npu-a2 (0) | 2.5min | 环境问题 | 下载sgl-kernel-npu依赖包时代理服务器返回500错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274174) |
| multimodal-gen-test-2-npu-a3 | 23.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传工件提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274177) |
| stage-b-test-2-npu-a2 (0) | 4.7min | 环境问题 | 自定义容器执行失败，测试进程异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274185) |
| stage-b-test-2-npu-a2 (1) | 2.8min | 环境问题 | 下载sgl-kernel-npu依赖时代理返回500错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274190) |
| single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms | 26.8min | 其他 | 日志被截断，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91514150190) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16, test/registered/ascend/acc... / glm5_top64_pruned_bf16_8p_gsm8k | 23.7min | 环境问题 | 作业在备份plog日志后正常结束，无测试失败信息，疑似环境或基础设施问题导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91524892500) |

- **stage-a-unit-test-npu**: CI在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载release包，但代理返回500 INTERNAL SERVER ERROR，导致依赖安装失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274164

- **stage-b-test-1-npu-a2 (0)**: CI在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载zip包，但代理返回500 INTERNAL SERVER ERROR，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274174

- **multimodal-gen-test-2-npu-a3**: 日志被截断，中间省略了关键测试输出。仅看到Node 20弃用警告和diffusion-failures目录无文件上传的提示，无法判断具体失败原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274177

- **stage-b-test-2-npu-a2 (0)**: 测试在运行test_npu_mla_fia_w8a8int8.py时，自定义容器实现执行失败，导致作业提前终止，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274185

- **stage-b-test-2-npu-a2 (1)**: 在安装依赖阶段，通过gh-proxy.test.osinfra.cn代理下载sgl-kernel-npu压缩包时，代理服务器返回500 INTERNAL SERVER ERROR，导致安装失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274190

- **single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms**: 提供的日志仅包含作业启动和清理阶段，未显示测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91514150190

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16, test/registered/ascend/acc... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在备份plog文件后进入清理阶段，未出现测试执行或断言失败信息，可能因NPU环境异常或资源问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91524892500

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274163) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274168) |
| multimodal-gen-test-1-npu-a3 | 34.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274170) |
| stage-b-test-1-npu-a2 (1) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274175) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274437) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 46.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274470) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274495) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274511) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274530) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91501274536) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91510537250) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91510890536) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91511047447) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91511097723) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91511790604) |
| single-node-poc (qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu... / qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91520554720) |
| single-node-poc (glm4_7_flash_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/glm4_... / glm4_7_flash_1p_gsm8k | 53.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91524187930) |
| single-node-poc (qwen3_vl_30b_a3b_bf16_2p_gsm8k, linux-aarch64-a3-4, test/registered/ascend/accur... / qwen3_vl_30b_a3b_bf16_2p_gsm8k | 47.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91524588448) |
| single-node-poc (moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/... / moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91524909740) |
| single-node-poc (qwen3_5_9b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/qw... / qwen3_5_9b_bf16_1p_gsm8k | 32.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30749612101/job/91531434792) |


## [Run #30748870559](https://github.com/sgl-project/sglang/actions/runs/30748870559)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 97.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30748870559

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 4.5min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266812) |
| multimodal-gen-test-1-npu-a3 | 2.7min | 其他 | 作业日志被截断，未显示实际失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266820) |
| multimodal-gen-test-2-npu-a3 | 23.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266823) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 2.8min | 环境问题 | 测试未生成metrics.json文件，导致性能测试无法完成。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499267047) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 33.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91505154921) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 29.8min | 环境问题 | Azure Blob 存储中的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91505548198) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 22.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91506258225) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91507895066) |

- **stage-b-test-4-npu-a3**: 作业在加载模型分片后，TP进程获取环境变量时出现错误，导致自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266812

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体错误。仅看到上传diffusion-failures目录时提示无文件，可能测试未产生失败产物或测试本身未执行成功，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266820

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266823

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 作业在运行性能测试时，未能找到/tmp/metrics.json文件，无法上传性能指标，测试提前结束。可能是测试脚本未正确执行或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499267047

- **single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明CI作业尝试访问的远程资源（如模型权重或测试数据）在存储中缺失，可能是资源被删除、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91505154921

- **single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16**: 作业尝试下载指定的 blob 日志文件，但返回 BlobNotFound 错误，说明该文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91505548198

- **single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明作业依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查 CI 配置中的 blob 引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91506258225

- **single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91507895066

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266808) |
| stage-b-test-1-npu-a2 (0) | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266810) |
| stage-b-test-16-npu-a3 | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266818) |
| stage-b-test-2-npu-a2 (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266825) |
| stage-b-test-1-npu-a2 (1) | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266833) |
| stage-b-test-2-npu-a2 (1) | 11.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499266834) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499267045) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499267046) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499267050) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 44.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499267064) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30748870559/job/91499267066) |


## [Run #30747340434](https://github.com/sgl-project/sglang/actions/runs/30747340434)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 42.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30747340434

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 41.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123595) |
| stage-b-test-4-npu-a3 | 41.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123597) |
| multimodal-gen-test-1-npu-a3 | 41.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123611) |
| multimodal-gen-test-2-npu-a3 | 41.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123635) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 41.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123876) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 41.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123909) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 41.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123938) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 41.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123963) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 41.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123997) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 41.3min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123999) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123595

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123597

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件可能已被删除、路径错误或未上传成功，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123611

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123635

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123876

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误信息为BlobNotFound，说明作业依赖的某个文件或资源在存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123909

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123938

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查资源是否存在及配置是否正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123963

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123997

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示BlobNotFound错误，可能是CI脚本尝试下载或访问的模型/数据文件在存储中缺失或路径错误，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123999

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123601) |
| stage-b-test-2-npu-a2 (0) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123607) |
| stage-b-test-1-npu-a2 (0) | 35.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123614) |
| stage-b-test-2-npu-a2 (1) | 10.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123630) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747340434/job/91495123651) |


## [Run #30747070165](https://github.com/sgl-project/sglang/actions/runs/30747070165)
- **分支**: `fix_hisparse_pd_dsv4`
- **总耗时**: 436.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30747070165

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 2.3min | 环境问题 | 下载sgl-kernel-npu依赖包时代理服务器返回500错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393362) |
| multimodal-gen-test-2-npu-a3 | 32.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393379) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 2.1min | 环境问题 | 作业在准备阶段即失败，未进入实际测试，日志显示缺少metrics.json文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393836) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 14.2min | 其他 | 日志被截断，未显示测试执行结果，无法判断失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393842) |
| single-node-poc (qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu... / qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms | 21.8min | 其他 | 日志被截断，未显示测试执行结果，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91510281735) |

- **stage-b-test-16-npu-a3**: 在安装依赖阶段，通过gh-proxy.test.osinfra.cn代理下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，代理服务器返回500 INTERNAL SERVER ERROR，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393362

- **multimodal-gen-test-2-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393379

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 作业在运行早期就终止，未执行测试逻辑。日志显示找不到/tmp/metrics.json，且plog备份步骤未找到plog文件，表明环境初始化或依赖安装阶段出现问题，导致测试无法启动。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393836

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志仅包含作业初始化和清理阶段，未展示测试运行输出或错误信息，可能因日志截断或作业在测试前被中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393842

- **single-node-poc (qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu... / qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms**: 日志仅包含作业启动、环境准备和plog备份步骤，未展示测试运行过程及失败点，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91510281735

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393364) |
| stage-b-test-1-npu-a2 (0) | 35.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393367) |
| stage-b-test-1-npu-a2 (1) | 33.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393368) |
| multimodal-gen-test-1-npu-a3 | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393391) |
| stage-a-unit-test-npu | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393415) |
| stage-b-test-2-npu-a2 (0) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393435) |
| stage-b-test-2-npu-a2 (1) | 10.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393443) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393776) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393821) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 19.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393850) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91494393882) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91500261203) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91500825318) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91501111069) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91501387489) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91501794823) |
| single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms | 26.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91501896069) |
| single-node-poc (glm4_7_flash_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/glm4_... / glm4_7_flash_1p_gsm8k | 56.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91510761720) |
| single-node-poc (qwen3_vl_30b_a3b_bf16_2p_gsm8k, linux-aarch64-a3-4, test/registered/ascend/accur... / qwen3_vl_30b_a3b_bf16_2p_gsm8k | 49.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91511813245) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16, test/registered/ascend/acc... / glm5_top64_pruned_bf16_8p_gsm8k | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91512209887) |
| single-node-poc (moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/... / moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91514969946) |
| single-node-poc (qwen3_5_9b_bf16_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/qw... / qwen3_5_9b_bf16_1p_gsm8k | 32.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30747070165/job/91515615338) |


---
*Auto-generated by npu_pr_monitor.py*