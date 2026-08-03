# NPU CI 执行监控
**生成时间**: 2026-08-03 12:19 UTC
**分析 Run 数**: 19

---

## [Run #30807539921](https://github.com/sgl-project/sglang/actions/runs/30807539921)
- **分支**: `fix/umbp-hicache-v4-hybrid-multipool`
- **总耗时**: 63.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30807539921

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398240) |
| stage-b-test-8-npu-a3 | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398250) |
| stage-b-test-4-npu-a3 (1) | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398253) |
| stage-b-test-1-npu-a2 (1) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398261) |
| stage-a-unit-test-npu | 29.0min | 环境问题 | 下载triton-ascend依赖时网络中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398277) |
| stage-b-test-4-npu-a3 (0) | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398287) |
| stage-b-test-1-npu-a3 | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398290) |
| stage-b-test-2-npu-a2 (0) | 34.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398292) |
| multimodal-gen-test-2-npu-a3 | 57.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398307) |
| multimodal-gen-test-1-npu-a3 | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398330) |
| stage-b-test-2-npu-a2 (1) | 3.4min | 环境问题 | K8s Pod 拉取镜像失败，ImagePullBackOff 导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398335) |
| stage-b-test-16-npu-a3 | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398355) |
| stage-b-test-1-npu-a2 (0) | 18.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398499) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398808) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398813) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398853) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 57.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398902) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 57.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398914) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398971) |

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398240

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398250

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或存储配置问题，属于环境依赖故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398253

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398261

- **stage-a-unit-test-npu**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中连接中断，重试后仍失败，最终导致作业执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398277

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到，可能是文件被删除、路径错误或上传失败，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398287

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398290

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398292

- **multimodal-gen-test-2-npu-a3**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398307

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的模型/数据文件在 Azure Blob 存储中缺失或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398330

- **stage-b-test-2-npu-a2 (1)**: 作业在启动阶段因容器镜像 swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-910b-ubuntu22.04-py3.11 拉取失败（ImagePullBackOff）而终止，可能因镜像不存在、凭据无效或网络问题，需检查镜像配置与节点连通性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398335

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398355

- **stage-b-test-1-npu-a2 (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于基础设施/环境问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398499

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398808

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败或配置变更，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398813

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398853

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中不存在，可能是文件被删除、路径错误或上传失败。这属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398902

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是日志保留策略或上传失败所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398914

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30807539921/job/91666398971


## [Run #30805956107](https://github.com/sgl-project/sglang/actions/runs/30805956107)
- **分支**: `lmzheng/startup-memory-observability`
- **总耗时**: 27.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30805956107

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661385931) |
| stage-b-test-1-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661385960) |
| stage-b-test-1-npu-a2 (1) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661385971) |
| stage-b-test-2-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661385982) |
| stage-b-test-4-npu-a3 (0) | 26.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386001) |
| stage-b-test-1-npu-a2 (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386004) |
| multimodal-gen-test-2-npu-a3 | 26.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386011) |
| stage-b-test-2-npu-a2 (1) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386012) |
| stage-b-test-8-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386018) |
| stage-a-unit-test-npu | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386032) |
| stage-b-test-4-npu-a3 (1) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386043) |
| stage-b-test-16-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386049) |
| multimodal-gen-test-1-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386105) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386716) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386730) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 26.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386751) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386777) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 26.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386796) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386815) |

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661385931

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661385960

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661385971

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置和依赖文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661385982

- **stage-b-test-4-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386001

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure 存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386004

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386011

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386012

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386018

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386032

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386043

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386049

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386105

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源缺失或路径错误，可能是文件被清理、上传失败或配置错误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386716

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386730

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 作业日志返回BlobNotFound错误，说明CI依赖的某个文件或资源在Azure Blob存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386751

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386777

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 作业在尝试访问Azure Blob存储时，返回BlobNotFound错误，说明所需的数据文件或资源未找到，可能是路径错误或文件未上传，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386796

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源是否存在或更新引用路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805956107/job/91661386815


## [Run #30805335702](https://github.com/sgl-project/sglang/actions/runs/30805335702)
- **分支**: `mmangkad/ci-rust-build-store`
- **总耗时**: 84.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30805335702

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 30.1min | 环境问题 | 下载triton-ascend依赖时自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931744) |
| stage-b-test-2-npu-a2 (1) | 26.5min | 环境问题 | 下载triton_ascend依赖时网络连接反复中断，导致下载失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931752) |
| stage-b-test-16-npu-a3 | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931761) |
| stage-b-test-4-npu-a3 (1) | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931774) |
| stage-b-test-4-npu-a3 (0) | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931788) |
| multimodal-gen-test-1-npu-a3 | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931792) |
| stage-a-unit-test-npu | 32.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931795) |
| stage-b-test-1-npu-a2 (1) | 28.6min | 环境问题 | 下载triton_ascend依赖时网络连接反复中断，导致下载失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931810) |
| stage-b-test-8-npu-a3 | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931819) |
| stage-b-test-2-npu-a3 | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931822) |
| multimodal-gen-test-2-npu-a3 | 79.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931824) |
| stage-b-test-1-npu-a3 | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931827) |
| stage-b-test-1-npu-a2 (0) | 31.1min | 环境问题 | 下载triton_ascend依赖时网络连接反复中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931894) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932465) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 79.0min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932510) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 79.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932541) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932585) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932636) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 79.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932658) |

- **stage-b-test-2-npu-a2 (0)**: 在安装triton-ascend==3.2.1.dev20260530（188.5MB）过程中，自定义容器实现执行失败，导致作业中断，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931744

- **stage-b-test-2-npu-a2 (1)**: 在安装triton_ascend-3.2.1.dev20260530-cp311-cp311-manylinux_2_27_aarch64.whl时，网络多次中断，下载进度缓慢（仅12MB/188.5MB），最终导致自定义容器执行失败，作业中止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931752

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931761

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931774

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931788

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931792

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931795

- **stage-b-test-1-npu-a2 (1)**: 在安装triton_ascend-3.2.1.dev20260530-cp311-cp311-manylinux_2_27_aarch64.whl时，网络多次中断，重试4次仅下载13.9MB/188.5MB，最终容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931810

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931819

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源被清理或配置变更，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931822

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业依赖的某个文件或数据在存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931824

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931827

- **stage-b-test-1-npu-a2 (0)**: 在安装triton_ascend-3.2.1.dev20260530-cp311-cp311-manylinux_2_27_aarch64.whl时，网络多次中断，下载速度极慢（约14kB/s），多次重试后仍无法完成，最终导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660931894

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932465

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示BlobNotFound错误，说明CI作业尝试下载的日志或数据文件在Azure Blob存储中已被删除或路径错误，属于环境配置或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932510

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储账户中缺失，可能是文件被误删、路径配置错误或上传未完成。需检查CI流程中依赖的blob路径及上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932541

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932585

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932636

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 作业日志返回BlobNotFound错误，表明CI流程尝试访问的Azure Blob存储资源缺失或路径错误，属于环境配置或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30805335702/job/91660932658


## [Run #30803966032](https://github.com/sgl-project/sglang/actions/runs/30803966032)
- **分支**: `codex/feat/pd-decode-fused-topk`
- **总耗时**: 77.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30803966032

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 13.2min | 环境问题 | 下载torch-npu依赖时超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071368) |
| stage-b-test-1-npu-a3 | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071371) |
| stage-b-test-1-npu-a2 (1) | 2.6min | 环境问题 | rustup 下载超时导致安装 Rust 失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071393) |
| stage-b-test-2-npu-a2 (1) | 18.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071404) |
| stage-b-test-4-npu-a3 (1) | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071414) |
| stage-b-test-8-npu-a3 | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071419) |
| stage-b-test-2-npu-a3 | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071425) |
| stage-b-test-4-npu-a3 (0) | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071444) |
| stage-b-test-2-npu-a2 (0) | 11.7min | 环境问题 | 下载triton-ascend依赖时网络中断导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071457) |
| stage-a-unit-test-npu | 17.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071476) |
| multimodal-gen-test-2-npu-a3 | 71.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071479) |
| stage-b-test-16-npu-a3 | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071484) |
| multimodal-gen-test-1-npu-a3 | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071529) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071957) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 71.7min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071989) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655072023) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655072039) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 71.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655072104) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 71.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655072162) |

- **stage-b-test-1-npu-a2 (0)**: 在安装torch-npu==2.10.0时，下载32.5MB的whl包耗时超过4分钟，最终触发HttpClient超时，导致自定义容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071368

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071371

- **stage-b-test-1-npu-a2 (1)**: CI 环境中未预装 Rust，脚本尝试通过内部缓存服务下载 rustup 时网络超时，无法获取 channel-rust-1.92.toml.sha256，最终安装失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071393

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071404

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071414

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071419

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071425

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071444

- **stage-b-test-2-npu-a2 (0)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中网络连接中断，重试后仍失败，最终导致自定义容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071457

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071476

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071479

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071484

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071529

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源是否存在或更新引用路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071957

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI作业依赖的远程日志文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655071989

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或测试数据）未找到，可能是路径错误或资源未上传，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655072023

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655072039

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个数据文件或模型权重在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655072104

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程存储对象缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源准备问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803966032/job/91655072162


## [Run #30803795975](https://github.com/sgl-project/sglang/actions/runs/30803795975)
- **分支**: `lmzheng/startup-memory-observability`
- **总耗时**: 31.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30803795975

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496048) |
| stage-a-unit-test-npu | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496050) |
| multimodal-gen-test-2-npu-a3 | 30.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496093) |
| multimodal-gen-test-1-npu-a3 | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496124) |
| stage-b-test-1-npu-a3 | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496129) |
| stage-b-test-16-npu-a3 | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496134) |
| stage-b-test-4-npu-a3 (0) | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496141) |
| stage-b-test-1-npu-a2 (0) | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496153) |
| stage-b-test-2-npu-a2 (0) | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496177) |
| stage-b-test-4-npu-a3 (1) | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496183) |
| stage-b-test-1-npu-a2 (1) | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496231) |
| stage-b-test-2-npu-a2 (1) | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496259) |
| stage-b-test-2-npu-a3 | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496288) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 30.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496892) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496893) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496947) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 30.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496954) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 30.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654497026) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 30.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654497033) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496048

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496050

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496093

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496124

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496129

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496134

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496141

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496153

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径和生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496177

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496183

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496231

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496259

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496288

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 作业在下载或访问Azure Blob存储中的资源时，返回BlobNotFound错误，说明相关文件缺失或路径错误，属于环境或资源准备问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496892

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储资源缺失或路径错误，可能是文件被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496893

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败或配置变更，需检查相关 blob 是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496947

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明作业依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查 CI 配置中的 blob 引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654496954

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明CI依赖的某个文件或资源未上传或已被删除，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654497026

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是日志保留策略或上传失败所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30803795975/job/91654497033


## [Run #30800573902](https://github.com/sgl-project/sglang/actions/runs/30800573902)
- **分支**: `lmzheng/startup-memory-observability`
- **总耗时**: 46.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30800573902

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072904) |
| stage-b-test-8-npu-a3 | 45.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072951) |
| stage-b-test-1-npu-a2 (0) | 28.6min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072970) |
| multimodal-gen-test-1-npu-a3 | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072986) |
| stage-b-test-4-npu-a3 (1) | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072988) |
| stage-b-test-16-npu-a3 | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073001) |
| stage-b-test-2-npu-a3 | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073010) |
| multimodal-gen-test-2-npu-a3 | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073019) |
| stage-b-test-1-npu-a2 (1) | 28.9min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073039) |
| stage-b-test-4-npu-a3 (0) | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073045) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073599) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073614) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073624) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 45.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073658) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 45.2min | 环境问题 | Azure Blob 存储中指定的模型权重文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073690) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 45.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073790) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072904

- **stage-b-test-8-npu-a3**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源已被删除或路径错误，可能是上游产物未上传或配置失效，需检查依赖的存储路径及上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072951

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU驱动、容器资源或配置异常导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072970

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072986

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure 存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072988

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073001

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073010

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073019

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行到80%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073039

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073045

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073599

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的 Azure Blob 存储资源缺失或路径错误，可能是数据未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073614

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073624

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073658

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的模型权重（qwen3_6_27b）在指定路径下缺失，可能是模型未上传或路径配置错误，属于环境/资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073690

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储账户中缺失，可能是文件被误删、路径配置错误或上传未完成。需检查CI流程中依赖的blob路径是否正确，并确认文件是否已成功上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073790

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072960) |
| stage-a-unit-test-npu | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644072968) |
| stage-b-test-2-npu-a2 (1) | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30800573902/job/91644073015) |


## [Run #30800233294](https://github.com/sgl-project/sglang/actions/runs/30800233294)
- **分支**: `lmzheng/startup-memory-observability`
- **总耗时**: 5.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30800233294

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276514) |
| stage-b-test-2-npu-a2 (0) | 3.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276588) |
| stage-a-unit-test-npu | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276589) |
| stage-b-test-4-npu-a3 (1) | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276593) |
| stage-b-test-2-npu-a2 (1) | 3.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276597) |
| stage-b-test-1-npu-a2 (1) | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276630) |
| multimodal-gen-test-2-npu-a3 | 3.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276648) |
| stage-b-test-1-npu-a2 (0) | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276668) |
| stage-b-test-16-npu-a3 | 3.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276670) |
| stage-b-test-4-npu-a3 (0) | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276694) |
| stage-b-test-2-npu-a3 | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276737) |
| stage-b-test-8-npu-a3 | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276757) |
| multimodal-gen-test-1-npu-a3 | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277000) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 3.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277393) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277418) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277481) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 3.1min | 环境问题 | Azure Blob 存储中指定的模型权重文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277558) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277559) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277588) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276514

- **stage-b-test-2-npu-a2 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276588

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276589

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276593

- **stage-b-test-2-npu-a2 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276597

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276630

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试下载的blob（可能为模型权重或测试数据）在存储账户中不存在，可能是文件被删除、路径错误或上传失败，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276648

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276668

- **stage-b-test-16-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据在存储中缺失，可能是上游作业未成功上传或路径配置错误，属于环境或依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276670

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276694

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276737

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643276757

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277000

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误信息为BlobNotFound，表明作业依赖的某个blob（可能是模型权重、测试数据或配置文件）在存储中缺失或路径错误，需检查相关资源是否已正确上传或路径配置是否正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277393

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是配置问题或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277418

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置变更，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277481

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的模型权重（可能为 qwen3_6_27b 相关文件）在 Azure Blob 存储中缺失或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277558

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277559

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91643277588

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 1.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30800233294/job/91642925520) |


## [Run #30799715867](https://github.com/sgl-project/sglang/actions/runs/30799715867)
- **分支**: `lmzheng/startup-memory-observability`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30799715867

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525801) |
| stage-b-test-2-npu-a2 (0) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525813) |
| stage-b-test-1-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525820) |
| stage-b-test-16-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525839) |
| stage-b-test-4-npu-a3 (1) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525846) |
| stage-b-test-2-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525850) |
| stage-b-test-2-npu-a2 (1) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525859) |
| stage-b-test-1-npu-a2 (0) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525860) |
| multimodal-gen-test-2-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525869) |
| stage-b-test-8-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525880) |
| stage-b-test-4-npu-a3 (0) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525887) |
| stage-b-test-1-npu-a2 (1) | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525911) |
| multimodal-gen-test-1-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525975) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526579) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526660) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526692) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526753) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526777) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 4.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526799) |

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525801

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525813

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525820

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525839

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525846

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525850

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525859

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525860

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525869

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525880

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525887

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问一个不存在的存储对象（如模型权重、缓存或日志），需检查相关路径或资源是否被误删或配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525911

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641525975

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的模型权重或数据文件未上传到指定存储位置，可能是文件缺失或路径配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526579

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或测试数据）在 Azure Blob 中缺失，可能是资源被删除或路径配置错误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526660

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是配置问题或数据未上传，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526692

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526753

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败或路径错误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526777

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被误删、路径配置错误或上传未完成。需检查CI配置中的blob路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641526799

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 1.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30799715867/job/91641266277) |


## [Run #30799075246](https://github.com/sgl-project/sglang/actions/runs/30799075246)
- **分支**: `lmzheng/startup-memory-observability`
- **总耗时**: 9.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30799075246

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575853) |
| stage-b-test-4-npu-a3 (1) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575859) |
| stage-b-test-1-npu-a2 (1) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575862) |
| stage-b-test-1-npu-a2 (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575881) |
| stage-b-test-2-npu-a2 (0) | 8.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575897) |
| stage-b-test-4-npu-a3 (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575900) |
| stage-b-test-2-npu-a2 (1) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575923) |
| multimodal-gen-test-1-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575961) |
| stage-b-test-2-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575962) |
| stage-b-test-16-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575965) |
| stage-b-test-8-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576002) |
| multimodal-gen-test-2-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576009) |
| stage-b-test-1-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576059) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576538) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576566) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576568) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576621) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576678) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576700) |

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575853

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本引用了不存在的存储路径或文件，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575859

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575862

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575881

- **stage-b-test-2-npu-a2 (0)**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575897

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575900

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成或存储配置有误，需检查相关 blob 路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575923

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或存储配置变更，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575961

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575962

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639575965

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576002

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576009

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576059

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576538

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或模型权重在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576566

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576568

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或数据在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576621

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576678

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明作业依赖的 Azure Blob 存储资源缺失或路径错误，可能是数据未上传、被删除或配置错误，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639576700

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 1.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30799075246/job/91639228131) |


## [Run #30798965399](https://github.com/sgl-project/sglang/actions/runs/30798965399)
- **分支**: `codex/feat/pd-decode-fused-topk`
- **总耗时**: 72.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30798965399

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (1) | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006485) |
| stage-b-test-2-npu-a3 | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006490) |
| stage-b-test-1-npu-a2 (0) | 44.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006531) |
| stage-b-test-16-npu-a3 | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006533) |
| stage-b-test-1-npu-a2 (1) | 43.5min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006543) |
| stage-b-test-1-npu-a3 | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006555) |
| stage-b-test-4-npu-a3 (0) | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006556) |
| multimodal-gen-test-1-npu-a3 | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006558) |
| multimodal-gen-test-2-npu-a3 | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006559) |
| stage-b-test-8-npu-a3 | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006584) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006944) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006961) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006967) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006983) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 71.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006986) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 71.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639007004) |

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储文件缺失或路径错误，可能是资源被清理或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006485

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006490

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006531

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006533

- **stage-b-test-1-npu-a2 (1)**: 日志显示sglang服务正常启动并处理请求，但随后出现'Executing the custom container implementation failed'错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006543

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 中不存在，可能是资源被清理、路径错误或上传失败，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006555

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的构建产物或缓存文件未上传或已被删除，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006556

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或数据在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006558

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006559

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006584

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件未上传，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006944

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006961

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006967

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006983

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006986

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题，需检查CI依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639007004

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006515) |
| stage-b-test-2-npu-a2 (1) | 13.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006530) |
| stage-a-unit-test-npu | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30798965399/job/91639006548) |


## [Run #30798721812](https://github.com/sgl-project/sglang/actions/runs/30798721812)
- **分支**: `mmangkad/deepseek-v4-0731-reasoning-effort`
- **总耗时**: 7.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30798721812

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248039) |
| stage-b-test-1-npu-a3 | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248128) |
| stage-b-test-2-npu-a2 (1) | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248141) |
| stage-b-test-1-npu-a2 (0) | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248144) |
| stage-b-test-8-npu-a3 | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248149) |
| multimodal-gen-test-2-npu-a3 | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248153) |
| multimodal-gen-test-1-npu-a3 | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248155) |
| stage-b-test-1-npu-a2 (1) | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248158) |
| stage-a-unit-test-npu | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248166) |
| stage-b-test-2-npu-a2 (0) | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248171) |
| stage-b-test-4-npu-a3 (1) | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248178) |
| stage-b-test-4-npu-a3 (0) | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248185) |
| stage-b-test-16-npu-a3 | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248207) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248809) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 6.9min | 环境问题 | 日志显示BlobNotFound错误，即所需资源文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248875) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248878) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248880) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248891) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 6.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248908) |

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248039

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的存储对象缺失或路径错误，可能是资源未上传或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248128

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248141

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248144

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248149

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248153

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248155

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248158

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248166

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或依赖文件在 Azure Blob 存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248171

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248178

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248185

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248207

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248809

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业在获取依赖的blob文件时失败，返回404错误，可能是资源被清理或路径配置错误，属于环境或配置问题，需检查资源是否存在及路径是否正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248875

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件或数据在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查 CI 配置中的 blob 路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248878

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是临时文件清理或配置问题，需检查存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248880

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248891

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798721812/job/91638248908


## [Run #30798312875](https://github.com/sgl-project/sglang/actions/runs/30798312875)
- **分支**: `feat/adaptive_spec/sglang/cost-aware-adaptive-sd`
- **总耗时**: 41.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30798312875

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 2.9min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139486) |
| stage-b-test-1-npu-a3 | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139503) |
| stage-b-test-2-npu-a2 (1) | 7.5min | 环境问题 | 自定义容器执行失败，torch-npu安装后容器异常退出 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139520) |
| stage-b-test-1-npu-a2 (0) | 5.9min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139526) |
| stage-b-test-1-npu-a2 (1) | 1.9min | 环境问题 | K8s Pod 拉取镜像失败，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139533) |
| stage-b-test-2-npu-a3 | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139541) |
| stage-b-test-4-npu-a3 (1) | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139545) |
| stage-b-test-2-npu-a2 (0) | 7.6min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139555) |
| stage-b-test-8-npu-a3 | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139562) |
| stage-b-test-16-npu-a3 | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139613) |
| stage-b-test-4-npu-a3 (0) | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139631) |
| multimodal-gen-test-1-npu-a3 | 39.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139635) |
| multimodal-gen-test-2-npu-a3 | 39.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139695) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637140773) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 39.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637140864) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 39.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637140955) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637141008) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637141096) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 39.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637141135) |

- **stage-a-unit-test-npu**: 日志显示在安装系统依赖包（如libgl1-mesa-glx等）后，出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139486

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139503

- **stage-b-test-2-npu-a2 (1)**: 日志显示在安装torch-npu==2.10.0后，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139520

- **stage-b-test-1-npu-a2 (0)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业中断。属于NPU环境依赖安装问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139526

- **stage-b-test-1-npu-a2 (1)**: 容器镜像 swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-910b-ubuntu22.04-py3.11 拉取失败（ImagePullBackOff），可能是镜像不存在、凭据无效或网络问题，需检查镜像仓库配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139533

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139541

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139545

- **stage-b-test-2-npu-a2 (0)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中，自定义容器实现执行失败，导致作业中断，属于NPU环境依赖安装问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139555

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139562

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或配置问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139613

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139631

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问日志时，Azure Blob 返回 BlobNotFound 错误，说明对应 blob 已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139635

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637139695

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637140773

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误信息为BlobNotFound，说明CI作业尝试访问的Azure Blob存储对象缺失或路径错误，可能是资源未上传、被删除或配置的URL有误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637140864

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业在尝试访问Azure Blob存储时遇到BlobNotFound错误，可能是由于文件被删除、路径错误或存储配置变更，导致CI流程无法继续执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637140955

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637141008

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637141096

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是测试数据或日志文件未正确上传，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91637141135

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 1.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30798312875/job/91636852793) |


## [Run #30797918423](https://github.com/sgl-project/sglang/actions/runs/30797918423)
- **分支**: `codex/feat/pd-decode-fused-topk`
- **总耗时**: 16.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30797918423

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731527) |
| stage-a-unit-test-npu | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731530) |
| stage-b-test-2-npu-a2 (1) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731541) |
| stage-b-test-2-npu-a2 (0) | 15.7min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731555) |
| stage-b-test-2-npu-a3 | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731566) |
| stage-b-test-1-npu-a2 (1) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731570) |
| stage-b-test-4-npu-a3 (0) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731581) |
| stage-b-test-16-npu-a3 | 15.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731608) |
| stage-b-test-8-npu-a3 | 15.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731624) |
| stage-b-test-4-npu-a3 (1) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731649) |
| stage-b-test-1-npu-a2 (0) | 15.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731684) |
| multimodal-gen-test-2-npu-a3 | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731699) |
| multimodal-gen-test-1-npu-a3 | 15.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731727) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732320) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732350) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 15.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732380) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732401) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 15.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732436) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732450) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在存储中缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731527

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储中找到，可能是资源被清理、路径错误或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731530

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731541

- **stage-b-test-2-npu-a2 (0)**: 日志显示BlobNotFound错误，说明CI作业尝试下载的日志或工件在Azure Blob存储中已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731555

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，需检查 blob 名称或恢复文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731566

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731570

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731581

- **stage-b-test-16-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731608

- **stage-b-test-8-npu-a3**: 错误码BlobNotFound表明CI依赖的远程存储对象缺失或路径错误，可能是上传失败、过期清理或配置变更所致，需检查相关存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731624

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731649

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731684

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731699

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635731727

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732320

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败或配置变更，需检查相关 blob 是否存在及路径配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732350

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查CI配置中的资源引用是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732380

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732401

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明CI作业尝试下载的远程资源（如模型权重或测试数据）已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732436

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797918423/job/91635732450


## [Run #30797549648](https://github.com/sgl-project/sglang/actions/runs/30797549648)
- **分支**: `mmangkad/fix-bcg-circular-import`
- **总耗时**: 20.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30797549648

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500297) |
| stage-b-test-2-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500299) |
| stage-b-test-1-npu-a2 (1) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500300) |
| stage-b-test-8-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500307) |
| stage-b-test-1-npu-a2 (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500324) |
| stage-b-test-2-npu-a2 (1) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500329) |
| stage-b-test-1-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500342) |
| stage-b-test-4-npu-a3 (1) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500355) |
| multimodal-gen-test-2-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500361) |
| multimodal-gen-test-1-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500382) |
| stage-a-unit-test-npu | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500388) |
| stage-b-test-4-npu-a3 (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500391) |
| stage-b-test-16-npu-a3 | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500400) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 19.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501188) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501237) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501288) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 19.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501294) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 19.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501309) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501366) |

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500297

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500299

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500300

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500307

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500324

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500329

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500342

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500355

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500361

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500382

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500388

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500391

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635500400

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被误删、路径配置错误或上传未完成。需检查CI流程中依赖的blob路径及上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501188

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501237

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除或路径错误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501288

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 错误码BlobNotFound表明CI依赖的远程存储资源缺失或路径错误，可能是文件被清理、上传失败或配置变更，需检查相关存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501294

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误信息为BlobNotFound，表明作业尝试访问的Azure Blob存储资源缺失或路径错误，可能是由于资源被删除、路径配置错误或网络问题导致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501309

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797549648/job/91635501366


## [Run #30797072071](https://github.com/sgl-project/sglang/actions/runs/30797072071)
- **分支**: `fix/umbp-hicache-v4-hybrid-multipool`
- **总耗时**: 155.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30797072071

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (0) | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253290) |
| stage-b-test-1-npu-a3 | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253314) |
| stage-b-test-8-npu-a3 | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253315) |
| stage-b-test-2-npu-a3 | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253325) |
| multimodal-gen-test-1-npu-a3 | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253343) |
| stage-b-test-4-npu-a3 (1) | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253358) |
| stage-b-test-16-npu-a3 | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253368) |
| multimodal-gen-test-2-npu-a3 | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253378) |
| stage-b-test-1-npu-a2 (1) | 26.4min | 超时 | NPU采样后端测试超时，1200秒内未完成。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253411) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 153.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253946) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253959) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253994) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 153.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253998) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 153.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633254022) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 153.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633254103) |

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253290

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253314

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253315

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253325

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已被删除或路径错误，属于环境配置或资源缺失问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253343

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253358

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253368

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253378

- **stage-b-test-1-npu-a2 (1)**: test_npu_sampling_backend.py中的test_mmlu测试执行超过1200秒被强制终止，测试期间解码吞吐量仅40 token/s，性能过低导致超时，可能为性能回归或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253411

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查CI配置中的资源引用是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253946

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253959

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253994

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被误删、路径错误或上传未完成。建议检查CI配置中的blob路径及上传步骤，确认文件存在且权限正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253998

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633254022

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重、数据集或缓存）已被删除或路径错误，需检查CI配置中的资源引用是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633254103

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 1.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91632968956) |
| stage-b-test-1-npu-a2 (0) | 59.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253301) |
| stage-a-unit-test-npu | 10.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253306) |
| stage-b-test-2-npu-a2 (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253313) |
| stage-b-test-2-npu-a2 (1) | 12.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30797072071/job/91633253349) |


## [Run #30796784156](https://github.com/sgl-project/sglang/actions/runs/30796784156)
- **分支**: `main`
- **总耗时**: 36.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30796784156

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381964) |
| multimodal-gen-test-1-npu-a3 | 34.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381982) |
| stage-b-test-4-npu-a3 (1) | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381985) |
| stage-b-test-1-npu-a2 (1) | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381986) |
| stage-b-test-2-npu-a3 | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381996) |
| stage-a-unit-test-npu | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382000) |
| stage-b-test-1-npu-a2 (0) | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382007) |
| stage-b-test-8-npu-a3 | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382013) |
| stage-b-test-4-npu-a3 (0) | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382021) |
| stage-b-test-16-npu-a3 | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382023) |
| multimodal-gen-test-2-npu-a3 | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382026) |
| stage-b-test-2-npu-a2 (1) | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382031) |
| stage-b-test-2-npu-a2 (0) | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382033) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 34.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382326) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382381) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382388) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382406) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 34.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382476) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 34.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382600) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是上游产物未上传或存储配置变更，需检查相关依赖文件是否生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381964

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储账户中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381982

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381985

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381986

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632381996

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382000

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382007

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382013

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的工件/缓存文件已被删除或路径错误，属于外部存储依赖问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382021

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382023

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382026

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382031

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382033

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业日志仅包含BlobNotFound错误，表明CI系统尝试下载或访问的存储对象缺失，可能是配置错误或文件未上传，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382326

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、文件被清理或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382381

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382388

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是临时文件被清理或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382406

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382476

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题，需检查CI配置中的blob路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632382600

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 1.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30796784156/job/91632120793) |


## [Run #30796007188](https://github.com/sgl-project/sglang/actions/runs/30796007188)
- **分支**: `fp8_kv_cache_rebase`
- **总耗时**: 18.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30796007188

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757306) |
| stage-b-test-1-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757315) |
| stage-b-test-16-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757329) |
| stage-b-test-1-npu-a2 (0) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757334) |
| multimodal-gen-test-1-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757340) |
| stage-b-test-2-npu-a2 (1) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757341) |
| stage-b-test-2-npu-a2 (0) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757342) |
| stage-b-test-1-npu-a2 (1) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757347) |
| stage-b-test-8-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757376) |
| multimodal-gen-test-2-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757387) |
| stage-b-test-4-npu-a3 (1) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757389) |
| stage-b-test-4-npu-a3 (0) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757398) |
| stage-b-test-2-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757441) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758086) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758110) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758151) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758172) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 17.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758208) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 17.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758428) |

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757306

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757315

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757329

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未完成或存储配置错误，需检查相关 blob 路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757334

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757340

- **stage-b-test-2-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757341

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未完成或存储配置错误，需检查相关 blob 路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757342

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757347

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757376

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757387

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757389

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757398

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629757441

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758086

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758110

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758151

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758172

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题，需检查CI脚本中的下载路径或资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758208

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或数据文件未正确上传或已被删除，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30796007188/job/91629758428


## [Run #30795672299](https://github.com/sgl-project/sglang/actions/runs/30795672299)
- **分支**: `amd_helios`
- **总耗时**: 116.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30795672299

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716252) |
| stage-b-test-4-npu-a3 (1) | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716257) |
| stage-b-test-2-npu-a3 | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716291) |
| multimodal-gen-test-2-npu-a3 | 115.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716307) |
| multimodal-gen-test-1-npu-a3 | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716318) |
| stage-b-test-8-npu-a3 | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716342) |
| stage-b-test-16-npu-a3 | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716344) |
| stage-b-test-4-npu-a3 (0) | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716508) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 115.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716751) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 115.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716790) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716802) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 115.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716804) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 115.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716888) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 115.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716941) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716252

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716257

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源缺失或路径错误，可能是上传失败、文件被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716291

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问某个blob时，返回BlobNotFound错误，说明该资源已被删除或路径错误，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716307

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716318

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716342

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716344

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的构建产物或缓存文件未上传或已被删除，需检查相关存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716508

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是文件被删除、上传失败或配置错误，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716751

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查CI配置中的资源引用是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716790

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在存储中不存在，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716802

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716804

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716888

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 作业在尝试访问Azure Blob存储时，返回BlobNotFound错误，说明所需的文件或资源未找到，可能是配置错误或文件被删除，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716941

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716243) |
| stage-b-test-2-npu-a2 (1) | 11.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716256) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716259) |
| stage-b-test-1-npu-a2 (1) | 33.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716269) |
| stage-b-test-1-npu-a2 (0) | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30795672299/job/91628716285) |


## [Run #30795492111](https://github.com/sgl-project/sglang/actions/runs/30795492111)
- **分支**: `migrate-pr-nightly-to-560t`
- **总耗时**: 58.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30795492111

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 36.0min | 环境问题 | 依赖下载超时导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386501) |
| stage-a-unit-test-npu | 11.5min | 环境问题 | 下载triton-ascend依赖时网络中断导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386520) |
| stage-b-test-8-npu-a3 | 35.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386523) |
| multimodal-gen-test-1-npu-a3 | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386529) |
| stage-b-test-2-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386532) |
| stage-b-test-4-npu-a3 (1) | 42.1min | 环境问题 | PyPI缓存服务超时导致torch包下载失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386543) |
| stage-b-test-1-npu-a3 | 4.7min | 环境问题 | K8s Pod 拉取镜像失败，导致作业无法启动 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386564) |
| stage-b-test-4-npu-a3 (0) | 36.0min | 环境问题 | 依赖下载超时导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386571) |
| multimodal-gen-test-2-npu-a3 | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386586) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 57.6min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387276) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2-, test/registered/npu/accuracy/... / qwen3_vl_8b_thinking_1p_mmmu | 5.0min | 环境问题 | K8s Pod 拉取镜像失败，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387354) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387357) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387376) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 57.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387382) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 57.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387401) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 46.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91636665122) |

- **stage-b-test-16-npu-a3**: 安装torchvision时，从内部PyPI缓存服务下载包超时，重试3次后失败，导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386501

- **stage-a-unit-test-npu**: 在安装triton-ascend-3.2.1.dev20260530时，下载188.5MB的whl包过程中连接中断，仅下载33.8MB后重试仍失败，最终自定义容器执行报错，属于网络/环境不稳定问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386520

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或依赖文件在 Azure Blob 存储中已被删除或路径错误，属于环境或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386523

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386529

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386532

- **stage-b-test-4-npu-a3 (1)**: 作业在安装torch-2.10.0时，从内部PyPI缓存服务（cache-service.nginx-pypi-cache.svc.cluster.local）下载元数据超时，重试3次后失败，属于网络/缓存服务环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386543

- **stage-b-test-1-npu-a3**: 容器镜像 swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-a3-ubuntu22.04-py3.11 拉取失败（ImagePullBackOff），可能是镜像不存在、凭据无效或网络问题，需检查镜像仓库配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386564

- **stage-b-test-4-npu-a3 (0)**: 在安装torchvision时，从内部PyPI缓存服务下载包超时，重试3次后仍失败，导致作业中断。属于网络或缓存服务临时故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386571

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634386586

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI作业尝试访问的Azure Blob存储资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387276

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2-, test/registered/npu/accuracy/... / qwen3_vl_8b_thinking_1p_mmmu**: Pod 因 ImagePullBackOff 无法拉取镜像 swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-a3-ubuntu22.04-py3.11，可能是镜像不存在、凭据无效或网络问题，作业在准备阶段即失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387354

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件（如模型权重或测试数据）在 Azure Blob 存储中缺失，可能是路径错误或文件未上传，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387357

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是配置问题或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387376

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查CI配置中的资源引用是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387382

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或缓存）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91634387401

- **single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明CI依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30795492111/job/91636665122


---
*Auto-generated by npu_pr_monitor.py*