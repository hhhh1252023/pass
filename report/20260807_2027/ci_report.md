# NPU CI 执行监控
**生成时间**: 2026-08-07 12:27 UTC
**分析 Run 数**: 34

---

## [Run #31174113795](https://github.com/sgl-project/sglang/actions/runs/31174113795)
- **分支**: `codex/mm-cuda-ipc-stream-sync`
- **总耗时**: 28.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31174113795

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 27.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413966) |
| stage-b-test-4-npu-a3 (0) | 27.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413970) |
| stage-b-test-8-npu-a3 | 27.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413977) |
| stage-b-test-16-npu-a3 | 27.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413980) |
| multimodal-gen-test-1-npu-a3 | 27.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413989) |
| stage-b-test-4-npu-a3 (1) | 27.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414013) |
| multimodal-gen-test-2-npu-a3 | 27.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414066) |
| stage-b-test-1-npu-a3 | 27.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414100) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414479) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 27.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414482) |

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径和生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413966

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413970

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413977

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于环境或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413980

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的模型/数据文件未上传或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852413989

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）未上传或路径错误，需检查存储配置或文件是否已正确发布。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414013

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传，或路径配置错误，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414066

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414100

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是数据未上传或已被删除，需检查存储配置或重新上传数据。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414479

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业日志返回BlobNotFound错误，说明CI流程尝试访问的远程存储对象缺失或路径错误，可能是配置问题或资源未正确上传，属于环境配置或依赖资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414482

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31174113795/job/92852414027) |


## [Run #31173961049](https://github.com/sgl-project/sglang/actions/runs/31173961049)
- **分支**: `codex/mm-fabric-transport`
- **总耗时**: 32.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31173961049

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 2.2min | 环境问题 | GitHub Actions 运行器环境异常，作业在初始化阶段即失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955430) |
| multimodal-gen-test-2-npu-a3 | 29.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955481) |
| multimodal-gen-test-1-npu-a3 | 29.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955483) |
| stage-b-test-4-npu-a3 (0) | 27.4min | 环境问题 | 自定义容器执行失败，NPU作业在模型加载后崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955496) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 3.9min | 环境问题 | 自托管runner的k8s pod未成功创建，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955956) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 29.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851956068) |

- **stage-b-test-16-npu-a3**: 作业在 checkout 后立即结束，无测试执行日志，仅有 Node.js 20 弃用警告和孤儿进程清理提示，疑似运行器或基础设施问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955430

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955481

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955483

- **stage-b-test-4-npu-a3 (0)**: 日志显示模型加载和初始化正常，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955496

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示下载actions/checkout超时，随后prepareJob未完成，执行脚本时报错'jobPod must be set'，说明runner的Kubernetes pod初始化失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955956

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或缓存文件未上传或已被删除，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851956068

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955424) |
| stage-b-test-2-npu-a3 | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955500) |
| stage-b-test-8-npu-a3 | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955503) |
| stage-b-test-4-npu-a3 (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955532) |
| stage-b-test-1-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173961049/job/92851955594) |


## [Run #31173159222](https://github.com/sgl-project/sglang/actions/runs/31173159222)
- **分支**: `codex/mm-qwen-owner-transport`
- **总耗时**: 6.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31173159222

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393938) |
| stage-b-test-1-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393958) |
| stage-b-test-16-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393965) |
| stage-b-test-4-npu-a3 (1) | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393972) |
| stage-b-test-4-npu-a3 (0) | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393976) |
| multimodal-gen-test-1-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393978) |
| stage-a-unit-test-npu | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393998) |
| multimodal-gen-test-2-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850394049) |
| stage-b-test-2-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850394158) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 0.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850394477) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850394562) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393938

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个 blob 文件已被删除或路径错误，可能是构建产物或测试数据缺失，需检查上传步骤或存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393958

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是上游产物未上传或过期，需检查存储配置及依赖资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393965

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393972

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393976

- **multimodal-gen-test-1-npu-a3**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393978

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的某个文件或依赖在 Azure Blob 存储中不存在，可能是资源被清理、路径错误或上传失败，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850393998

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850394049

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850394158

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 错误码BlobNotFound表明作业尝试访问的模型或数据文件在Azure Blob存储中缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850394477

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173159222/job/92850394562


## [Run #31173148325](https://github.com/sgl-project/sglang/actions/runs/31173148325)
- **分支**: `codex/mm-cuda-ipc-stream-sync`
- **总耗时**: 6.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31173148325

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367870) |
| stage-b-test-4-npu-a3 (1) | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367907) |
| stage-b-test-2-npu-a3 | 0.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367915) |
| stage-b-test-1-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367928) |
| stage-a-unit-test-npu | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367959) |
| multimodal-gen-test-1-npu-a3 | 0.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367967) |
| stage-b-test-8-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367968) |
| stage-b-test-4-npu-a3 (0) | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367969) |
| multimodal-gen-test-2-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850368054) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850368240) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850368271) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367870

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367907

- **stage-b-test-2-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367915

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367928

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367959

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367967

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367968

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850367969

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850368054

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850368240

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或数据集）已被删除或路径错误，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173148325/job/92850368271


## [Run #31173141410](https://github.com/sgl-project/sglang/actions/runs/31173141410)
- **分支**: `codex/mm-fabric-transport`
- **总耗时**: 12.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31173141410

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 5.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583824) |
| multimodal-gen-test-1-npu-a3 | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583836) |
| stage-b-test-2-npu-a3 | 5.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583840) |
| stage-a-unit-test-npu | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583868) |
| stage-b-test-8-npu-a3 | 6.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583875) |
| stage-b-test-16-npu-a3 | 5.8min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583877) |
| stage-b-test-4-npu-a3 (1) | 5.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化或资源分配异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583908) |
| stage-b-test-4-npu-a3 (0) | 6.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583915) |
| multimodal-gen-test-2-npu-a3 | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583938) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849584411) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849584442) |

- **stage-b-test-1-npu-a3**: 日志显示服务启动成功且生成请求返回200，但随后自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583824

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583836

- **stage-b-test-2-npu-a3**: 作业在加载DeepseekV2模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583840

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583868

- **stage-b-test-8-npu-a3**: 作业在加载模型分片后，执行自定义容器时失败，错误为'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583875

- **stage-b-test-16-npu-a3**: 作业在加载模型分片（约88%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583877

- **stage-b-test-4-npu-a3 (1)**: 日志显示模型权重加载成功（30.31GB内存），但随后自定义容器实现执行失败，提示联系自托管runner管理员，可能是NPU设备或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583908

- **stage-b-test-4-npu-a3 (0)**: 日志显示在捕获批次过程中，自定义容器实现执行失败，提示联系自托管 runner 管理员，属于基础设施或容器环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583915

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的模型/数据文件未上传或路径错误，属于外部存储依赖问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849583938

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是配置问题或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849584411

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传或已被删除，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173141410/job/92849584442


## [Run #31173134023](https://github.com/sgl-project/sglang/actions/runs/31173134023)
- **分支**: `codex/mm-cpu-tensor-broadcast`
- **总耗时**: 45.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31173134023

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 40.7min | 环境问题 | NPU测试环境启动失败，服务端口连接被拒绝 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491560) |
| multimodal-gen-test-2-npu-a3 | 7.1min | 环境问题 | GitHub Actions 下载依赖超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491599) |
| multimodal-gen-test-1-npu-a3 | 9.0min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491689) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 14.1min | 其他 | 日志被截断，未显示实际测试结果，无法判断失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849492070) |

- **stage-b-test-16-npu-a3**: 测试作业在启动阶段失败，bootstrap服务(端口8996)和健康检查(端口11100)均无法连接，提示Connection refused，说明NPU环境或服务未正确启动，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491560

- **multimodal-gen-test-2-npu-a3**: 作业在准备阶段下载 actions/upload-artifact 时，因网络请求超过100秒超时被取消，多次重试仍失败，属于网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491599

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，仅显示上传diffusion-failures目录时无文件，未包含测试执行的具体错误或失败断言，无法判断是性能、精度还是环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491689

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 提供的日志仅包含作业启动、环境准备和收尾清理信息，未包含测试执行过程或错误输出，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849492070

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 10.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491529) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491533) |
| stage-b-test-1-npu-a3 | 26.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491553) |
| stage-b-test-2-npu-a3 | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491558) |
| stage-b-test-4-npu-a3 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491568) |
| stage-b-test-4-npu-a3 (0) | 32.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849491605) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 12.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31173134023/job/92849492105) |


## [Run #31172444914](https://github.com/sgl-project/sglang/actions/runs/31172444914)
- **分支**: `main`
- **总耗时**: 21.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31172444914

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 10.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31172444914/job/92847121551) |
| multimodal-gen-test-1-npu-a3 | 18.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31172444914/job/92847121560) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31172444914/job/92847121551

- **multimodal-gen-test-1-npu-a3**: 日志中仅显示Node.js 20弃用警告、上传diffusion-failures目录无文件等提示，未包含测试执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31172444914/job/92847121560


## [Run #31171909758](https://github.com/sgl-project/sglang/actions/runs/31171909758)
- **分支**: `codex/mm-cpu-tensor-broadcast`
- **总耗时**: 19.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31171909758

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 9.1min | 代码错误 | NPU PD分离测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994739) |
| stage-b-test-4-npu-a3 (0) | 15.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994759) |
| stage-b-test-1-npu-a3 | 14.5min | 环境问题 | Rust工具链安装过程中下载组件失败，导致自定义容器执行中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994767) |
| stage-b-test-8-npu-a3 | 15.9min | 环境问题 | 下载triton-ascend依赖时网络中断导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994780) |
| stage-b-test-4-npu-a3 (1) | 12.6min | 环境问题 | 自托管runner执行自定义容器实现失败，下载过程中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994782) |
| stage-b-test-2-npu-a3 | 14.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994803) |
| stage-a-unit-test-npu | 15.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994809) |
| multimodal-gen-test-2-npu-a3 | 15.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994837) |
| multimodal-gen-test-1-npu-a3 | 16.4min | 环境问题 | GitHub Actions 下载 upload-artifact 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994864) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.5min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845995278) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 14.9min | 其他 | 日志被截断，未显示实际测试结果，无法确定失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845995284) |

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py 测试在326秒后报错，返回退出码1，导致整个测试阶段0/6通过。具体错误信息未在日志中显示，但可判断为测试用例本身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994739

- **stage-b-test-4-npu-a3 (0)**: 日志显示在测试进行中（Capturing batches阶段）突然报错"Executing the custom container implementation failed"，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994759

- **stage-b-test-1-npu-a3**: 作业在安装Rust 1.92时，下载cargo和clippy组件后，自定义容器实现执行失败（可能是网络或缓存服务问题），导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994767

- **stage-b-test-8-npu-a3**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中网络连接中断，重试后仍失败，最终导致自定义容器执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994780

- **stage-b-test-4-npu-a3 (1)**: 日志显示在下载文件（约22%进度）时，runner报错“Executing the custom container implementation failed”，属于自托管runner环境或网络问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994782

- **stage-b-test-2-npu-a3**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994803

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994809

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994837

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功下载，属于网络或 GitHub 服务问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845994864

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业在准备阶段下载 actions/upload-artifact@v4 时，因网络问题连续三次超时（每次100秒），最终下载失败，导致作业无法启动。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845995278

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业启动和清理信息，未展示测试执行过程或错误输出，可能因日志截断或作业在早期阶段被中断，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171909758/job/92845995284


## [Run #31171888216](https://github.com/sgl-project/sglang/actions/runs/31171888216)
- **分支**: `codex/mm-qwen-owner-transport`
- **总耗时**: 24.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31171888216

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 16.5min | 环境问题 | 自托管runner执行自定义容器实现失败，下载过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719872) |
| multimodal-gen-test-1-npu-a3 | 21.9min | 环境问题 | GitHub Actions 运行环境 Node.js 20 弃用警告，但作业实际失败原因未在日志中明确显示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719878) |
| multimodal-gen-test-2-npu-a3 | 16.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719891) |
| stage-b-test-1-npu-a3 | 16.9min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719906) |
| stage-b-test-4-npu-a3 (1) | 15.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719933) |
| stage-b-test-4-npu-a3 (0) | 16.5min | 环境问题 | 自定义容器执行失败，apt源更新超时导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719938) |
| stage-b-test-8-npu-a3 | 17.1min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719943) |
| stage-a-unit-test-npu | 17.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719949) |
| stage-b-test-16-npu-a3 | 16.5min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719972) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.8min | 其他 | 测试未生成metrics.json文件，导致性能测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845720366) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 16.5min | 环境问题 | 作业在准备阶段被中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845720370) |

- **stage-b-test-2-npu-a3**: 作业在下载文件时（约38%进度）出现网络中断，导致自定义容器实现执行失败，属于runner环境或网络问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719872

- **multimodal-gen-test-1-npu-a3**: 日志显示 Node.js 20 弃用警告，但未出现明确错误或测试失败信息。作业可能在后续步骤中因环境问题（如依赖安装失败或资源不足）而中断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719878

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node 20弃用警告和上传artifact时未找到diffusion-failures目录，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719891

- **stage-b-test-1-npu-a3**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业中断，属于环境或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719906

- **stage-b-test-4-npu-a3 (1)**: 作业在启动测试时，自定义容器实现执行失败，错误提示联系自托管runner管理员，属于NPU测试环境基础设施问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719933

- **stage-b-test-4-npu-a3 (0)**: 日志显示在apt-get更新ubuntu-ports源时，jammy-updates InRelease下载超时（Ign:2），随后报错"Executing the custom container implementation failed"，属于自托管runner环境网络或缓存服务问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719938

- **stage-b-test-8-npu-a3**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包后，自定义容器实现执行失败，导致作业中断，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719943

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719949

- **stage-b-test-16-npu-a3**: 作业在加载模型分片（约48%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845719972

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业运行后未找到/tmp/metrics.json文件，无法上传性能指标，测试流程提前结束，可能因性能测试脚本未正确执行或输出路径错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845720366

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在备份plog文件后即进入清理阶段，未执行测试命令，可能因runner环境异常或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171888216/job/92845720370


## [Run #31171872798](https://github.com/sgl-project/sglang/actions/runs/31171872798)
- **分支**: `codex/mm-cuda-ipc-stream-sync`
- **总耗时**: 24.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31171872798

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 13.3min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367265) |
| stage-a-unit-test-npu | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367281) |
| stage-b-test-1-npu-a3 | 14.7min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367285) |
| stage-b-test-8-npu-a3 | 9.4min | 环境问题 | 作业在checkout后未执行测试即结束，疑似环境初始化或资源分配失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367295) |
| stage-b-test-4-npu-a3 (0) | 9.2min | 环境问题 | 作业在checkout后立即结束，未执行实际测试，疑似runner环境异常或作业被提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367331) |
| stage-b-test-4-npu-a3 (1) | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367333) |
| multimodal-gen-test-2-npu-a3 | 6.1min | 环境问题 | GitHub Actions 下载 upload-artifact 超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367340) |
| multimodal-gen-test-1-npu-a3 | 9.8min | 环境问题 | GitHub Actions 下载 upload-artifact 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367357) |
| stage-b-test-2-npu-a3 | 16.0min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367365) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846368892) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.4min | 环境问题 | 作业在启动阶段即失败，未执行实际测试，日志显示缺少metrics.json文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846368934) |

- **stage-b-test-16-npu-a3**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业中断，属于环境或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367265

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367281

- **stage-b-test-1-npu-a3**: 日志显示在模型初始化阶段（TokenizerManager初始化后）出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367285

- **stage-b-test-8-npu-a3**: 日志显示checkout成功，但随后仅清理进程并警告Node 20弃用，无测试执行记录，可能因NPU资源不足或runner环境异常导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367295

- **stage-b-test-4-npu-a3 (0)**: 日志显示checkout成功（HEAD为890558e），但随后仅清理进程并警告Node 20弃用，无测试输出，作业非正常结束，可能因runner资源问题或外部中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367331

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问不存在的存储对象，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367333

- **multimodal-gen-test-2-npu-a3**: 作业在下载 actions/upload-artifact@v4 时连续三次超时（100秒），导致无法获取该 action，作业在准备阶段即失败，未进入实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367340

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，虽然后续重试成功，但该网络问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367357

- **stage-b-test-2-npu-a3**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中，自定义容器实现执行失败，导致作业中断。可能是网络或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846367365

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或数据文件未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846368892

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 作业在准备阶段就终止，未运行测试用例，且提示未找到/tmp/metrics.json，可能是环境初始化或资源分配问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171872798/job/92846368934


## [Run #31171863471](https://github.com/sgl-project/sglang/actions/runs/31171863471)
- **分支**: `online-nvfp4-to-mxfp4-convert`
- **总耗时**: 55.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31171863471

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 49.4min | 代码错误 | NPU PD分离测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315059) |
| multimodal-gen-test-2-npu-a3 | 43.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315120) |

- **stage-b-test-16-npu-a3**: 测试文件test_npu_pd_disaggregation.py执行失败（exit code 1），其余5个测试均通过。该测试涉及PD分离功能，可能是代码逻辑或环境配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315059

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315120

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a3 | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315042) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315068) |
| stage-b-test-8-npu-a3 | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315070) |
| stage-b-test-4-npu-a3 (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315083) |
| multimodal-gen-test-1-npu-a3 | 53.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315092) |
| stage-b-test-4-npu-a3 (1) | 13.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315097) |
| stage-b-test-1-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315098) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 32.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315376) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171863471/job/92845315397) |


## [Run #31171849446](https://github.com/sgl-project/sglang/actions/runs/31171849446)
- **分支**: `codex/mm-fabric-transport`
- **总耗时**: 21.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31171849446

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 13.7min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304589) |
| stage-b-test-2-npu-a3 | 13.4min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304631) |
| stage-b-test-4-npu-a3 (0) | 13.1min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304632) |
| stage-b-test-4-npu-a3 (1) | 13.1min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304639) |
| stage-a-unit-test-npu | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304650) |
| stage-b-test-16-npu-a3 | 13.4min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304667) |
| multimodal-gen-test-1-npu-a3 | 14.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304683) |
| multimodal-gen-test-2-npu-a3 | 14.0min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304724) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 13.3min | 其他 | 日志被截断，未显示测试执行结果，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846305092) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 13.4min | 其他 | 日志被截断，未显示实际测试结果，无法确定失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846305097) |

- **stage-b-test-1-npu-a3**: 作业在加载模型权重（75%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304589

- **stage-b-test-2-npu-a3**: 日志显示模型权重加载成功（Qwen3MoeForCausalLM），但随后自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304631

- **stage-b-test-4-npu-a3 (0)**: 日志显示测试正在运行DPAttentionDP2TP2.test_regex_generate_phone时，runner报错"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304632

- **stage-b-test-4-npu-a3 (1)**: 作业在加载模型分片至82%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304639

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败、清理或配置问题，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304650

- **stage-b-test-16-npu-a3**: 作业在TP/EP多进程加载权重阶段，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304667

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304683

- **multimodal-gen-test-2-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，虽然后续重试成功，但可能影响作业稳定性。此外，Node 20 弃用警告和 diffusion-failures 目录无文件上传，均非根本原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304724

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志仅包含作业启动、环境准备和清理阶段，未包含测试执行及失败断言信息，无法定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846305092

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业初始化和清理过程，未展示测试执行部分，无法判断是精度、性能还是环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846305097

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31171849446/job/92846304635) |


## [Run #31170998811](https://github.com/sgl-project/sglang/actions/runs/31170998811)
- **分支**: `codex/mm-qwen-owner-transport`
- **总耗时**: 14.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31170998811

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 8.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696441) |
| multimodal-gen-test-2-npu-a3 | 12.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696467) |
| stage-b-test-2-npu-a3 | 4.5min | 环境问题 | 下载triton-ascend依赖时自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696475) |
| multimodal-gen-test-1-npu-a3 | 13.3min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696477) |
| stage-b-test-4-npu-a3 (0) | 5.0min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696486) |
| stage-a-unit-test-npu | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696488) |
| stage-b-test-4-npu-a3 (1) | 3.7min | 环境问题 | GitHub Actions 下载 actions/checkout 超时，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696493) |
| stage-b-test-1-npu-a3 | 6.1min | 环境问题 | GitHub Actions 下载 actions/checkout 时网络超时，但重试后成功。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696522) |
| stage-b-test-8-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696569) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 8.2min | 环境问题 | 作业在启动后立即被清理，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696858) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.5min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696874) |

- **stage-b-test-16-npu-a3**: 作业在启动NPU推理服务时，KV cache分配和Gloo通信初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696441

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败产物，但作业最终失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696467

- **stage-b-test-2-npu-a3**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中，自定义容器实现执行失败，导致作业中断。可能是网络或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696475

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功下载，属于网络/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696477

- **stage-b-test-4-npu-a3 (0)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中，自定义容器实现执行失败，导致作业中断，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696486

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败、清理或配置问题，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696488

- **stage-b-test-4-npu-a3 (1)**: 日志显示下载 actions/checkout 时请求超时（100秒），重试后仍失败，最终报错“Executing the custom container implementation failed”，属于网络或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696493

- **stage-b-test-1-npu-a3**: 作业在准备阶段下载 actions/checkout@v4 时首次请求超时（HTTP 请求超时 1 分 40 秒），重试后成功获取。后续流程正常完成，无其他错误，属于临时网络波动导致的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696522

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696569

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试命令，可能是runner环境异常或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696858

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业在准备阶段下载 actions/upload-artifact@v4 时，因网络问题多次超时（100秒），重试3次后仍失败，导致作业无法启动。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170998811/job/92842696874


## [Run #31170976709](https://github.com/sgl-project/sglang/actions/runs/31170976709)
- **分支**: `codex/mm-cuda-ipc-stream-sync`
- **总耗时**: 18.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31170976709

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 6.3min | 环境问题 | 自定义容器执行失败，下载triton-ascend依赖时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649104) |
| stage-b-test-1-npu-a3 | 6.0min | 环境问题 | 自定义容器执行失败，下载文件时网络中断导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649129) |
| stage-b-test-4-npu-a3 (1) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649135) |
| stage-b-test-4-npu-a3 (0) | 5.1min | 环境问题 | 自托管runner在下载依赖时容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649151) |
| stage-a-unit-test-npu | 7.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649160) |
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649164) |
| stage-b-test-16-npu-a3 | 8.0min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649166) |
| stage-b-test-8-npu-a3 | 7.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649182) |
| multimodal-gen-test-2-npu-a3 | 3.1min | 环境问题 | GitHub Actions 下载 upload-artifact 超时导致任务取消 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649280) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 8.1min | 其他 | 日志不完整，未显示测试执行结果，无法确定失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649746) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.0min | 环境问题 | 下载actions/upload-artifact超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649830) |

- **stage-b-test-2-npu-a3**: 作业在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中，自定义容器实现执行失败，导致作业终止。可能是网络或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649104

- **stage-b-test-1-npu-a3**: 日志显示在下载文件过程中（约4550K处）出现网络中断，随后报错"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649129

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649135

- **stage-b-test-4-npu-a3 (0)**: 日志显示在下载文件（约4600K处）时，自定义容器实现执行失败，错误为'Executing the custom container implementation failed'，属于runner环境或网络问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649151

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649160

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位真实原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649164

- **stage-b-test-16-npu-a3**: 日志显示模型分片加载过程中，DP0 TP0获取ASCEND_OPP_PATH后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649166

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明对应的 blob 已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649182

- **multimodal-gen-test-2-npu-a3**: 作业在准备阶段下载 actions/upload-artifact 时，因 HttpClient.Timeout 100秒限制导致请求取消，重试后仍失败，最终任务被取消，属于网络/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649280

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业初始化和清理阶段，未展示测试运行过程及错误信息，可能因日志截断或作业在启动阶段即被终止，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649746

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: GitHub Actions在下载upload-artifact时因网络超时（100秒）失败，重试后仍未能完成，导致后续步骤无法正常执行，最终作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170976709/job/92843649830


## [Run #31170959336](https://github.com/sgl-project/sglang/actions/runs/31170959336)
- **分支**: `codex/mm-fabric-transport`
- **总耗时**: 18.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31170959336

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 7.6min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053101) |
| stage-a-unit-test-npu | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053113) |
| stage-b-test-8-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053141) |
| stage-b-test-1-npu-a3 | 7.1min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053146) |
| stage-b-test-4-npu-a3 (1) | 3.5min | 环境问题 | 作业在准备阶段被中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053161) |
| stage-b-test-4-npu-a3 (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053169) |
| multimodal-gen-test-2-npu-a3 | 11.7min | 环境问题 | GitHub Actions 下载 actions/upload-artifact 时网络超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053172) |
| multimodal-gen-test-1-npu-a3 | 14.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053198) |
| stage-b-test-2-npu-a3 | 7.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053203) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 7.7min | 环境问题 | 作业在准备阶段即被终止，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053689) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.1min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053713) |

- **stage-b-test-16-npu-a3**: 作业在加载模型分片至74%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053101

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053113

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053141

- **stage-b-test-1-npu-a3**: 作业在加载权重完成后，自定义容器实现执行失败，提示联系自托管runner管理员，可能是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053146

- **stage-b-test-4-npu-a3 (1)**: 日志显示作业在checkout完成后，执行k8s/index.js时被清理（Cleaning up orphan processes），可能是runner被回收或作业被取消，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053161

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053169

- **multimodal-gen-test-2-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时多次因 HttpClient.Timeout 100秒超时而失败，最终作业未能正常完成。这是网络或 GitHub 服务问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053172

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到。可能是日志上传或下载路径错误，或文件已被删除，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053198

- **stage-b-test-2-npu-a3**: 作业在运行TestMoreRunnerBackendTritonDefault.test_moe_runner_backend测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053203

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在下载actions后立即进入清理阶段，未执行任何测试命令，可能因runner环境异常或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053689

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功，最终作业中断。属于网络或 GitHub 服务问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170959336/job/92843053713


## [Run #31170787625](https://github.com/sgl-project/sglang/actions/runs/31170787625)
- **分支**: `prb2-ltx2-bcg`
- **总耗时**: 55.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31170787625

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 49.9min | 环境问题 | GitHub Actions 下载 checkout action 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170787625/job/92841980984) |

- **multimodal-gen-test-2-npu-a3**: 下载 actions/checkout@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功下载，属于网络/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170787625/job/92841980984

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 54.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31170787625/job/92841980979) |


## [Run #31170645302](https://github.com/sgl-project/sglang/actions/runs/31170645302)
- **分支**: `codex/mm-cpu-tensor-broadcast`
- **总耗时**: 21.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31170645302

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 7.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523704) |
| stage-b-test-4-npu-a3 (0) | 6.3min | 环境问题 | 下载torch-npu依赖包时超时，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523744) |
| multimodal-gen-test-1-npu-a3 | 19.0min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523755) |
| stage-b-test-16-npu-a3 | 8.5min | 环境问题 | 自定义容器执行失败，模型分片加载完成后runner报错退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523775) |
| stage-b-test-4-npu-a3 (1) | 4.2min | 环境问题 | 下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523782) |
| multimodal-gen-test-2-npu-a3 | 21.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523832) |
| stage-b-test-2-npu-a3 | 7.9min | 代码错误 | 导入错误导致测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523964) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 18.3min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841524228) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 8.6min | 其他 | 日志不完整，未显示测试执行结果，作业在准备阶段后即结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841524268) |

- **stage-b-test-1-npu-a3**: 日志显示在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是容器环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523704

- **stage-b-test-4-npu-a3 (0)**: 在安装torch-npu==2.10.0时，下载32.5MB的whl包耗时超过1分钟，最终触发HTTP请求超时，导致作业失败。可能是网络波动或镜像源不稳定。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523744

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后作业继续但最终未上传产物，属于网络/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523755

- **stage-b-test-16-npu-a3**: 日志显示模型88个分片加载完成后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU环境或容器配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523775

- **stage-b-test-4-npu-a3 (1)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业中断，属于环境或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523782

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含multimodal-gen-test-2-npu-a3实际测试执行和失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523832

- **stage-b-test-2-npu-a3**: 日志显示无法从sglang.srt.managers.mm_utils导入MultimodalDataItem，说明代码中存在导入错误，导致测试进程崩溃。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523964

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示下载 actions/upload-artifact@v4 时两次因 HttpClient.Timeout 100秒超时而失败，重试后仍失败，属于网络/环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841524228

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业初始化和环境准备信息，未展示实际测试运行或失败原因，可能因日志截断或作业被提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841524268

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31170645302/job/92841523685) |


## [Run #31170630232](https://github.com/sgl-project/sglang/actions/runs/31170630232)
- **分支**: `codex/mm-cuda-ipc-stream-sync`
- **总耗时**: 10.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31170630232

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 1.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151137) |
| stage-a-unit-test-npu | 1.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151141) |
| stage-b-test-1-npu-a3 | 1.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151171) |
| multimodal-gen-test-1-npu-a3 | 3.7min | 超时 | GitHub Actions 下载依赖超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151178) |
| stage-b-test-4-npu-a3 (0) | 1.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151185) |
| stage-b-test-16-npu-a3 | 1.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151199) |
| multimodal-gen-test-2-npu-a3 | 6.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151214) |
| stage-b-test-4-npu-a3 (1) | 1.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151217) |
| stage-b-test-2-npu-a3 | 1.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151235) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 0.9min | 超时 | 作业在准备阶段被取消，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151409) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 1.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151611) |

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施/环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151137

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151141

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151171

- **multimodal-gen-test-1-npu-a3**: 作业在下载 actions/checkout 和 upload-artifact 后，因 HttpClient 100秒超时被取消，属于网络或基础设施问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151178

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151185

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151199

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置变更，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151214

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151217

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151235

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示在下载actions/checkout和upload-artifact后，作业被取消（The operation was canceled），可能是由于等待资源或网络问题导致超时，未开始运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151409

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170630232/job/92842151611


## [Run #31170623082](https://github.com/sgl-project/sglang/actions/runs/31170623082)
- **分支**: `codex/mm-fabric-transport`
- **总耗时**: 7.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31170623082

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 4.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576626) |
| stage-b-test-8-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576627) |
| multimodal-gen-test-2-npu-a3 | 4.4min | 环境问题 | 作业在准备阶段即被终止，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576636) |
| stage-b-test-16-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576637) |
| stage-b-test-2-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576641) |
| stage-b-test-1-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576643) |
| multimodal-gen-test-1-npu-a3 | 4.3min | 超时 | GitHub Actions 下载依赖时网络超时 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576687) |
| stage-b-test-4-npu-a3 (0) | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576701) |
| stage-b-test-4-npu-a3 (1) | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576739) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 4.2min | 环境问题 | Azure Blob 存储中指定的模型权重文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841577077) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 4.3min | 环境问题 | 作业在准备阶段被中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841577101) |

- **stage-a-unit-test-npu**: 日志显示在下载torch等依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576626

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576627

- **multimodal-gen-test-2-npu-a3**: 日志显示GitHub Actions在checkout完成后，运行k8s/index.js时进程被清理（Cleaning up orphan processes），作业提前结束，未执行任何多模态生成测试，疑似runner环境或调度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576636

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576637

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576641

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576643

- **multimodal-gen-test-1-npu-a3**: 作业在下载 actions/checkout 和 upload-artifact 后，因 HttpClient 100秒超时被取消，属于网络或基础设施问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576687

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576701

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841576739

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明模型权重（glm5_top64_pruned_bf16）在存储中缺失或路径错误，可能是上传失败或配置错误，需检查存储路径及文件是否完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841577077

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在checkout后运行自定义k8s脚本时被终止，可能是runner环境或调度问题，未出现测试执行或失败信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170623082/job/92841577101


## [Run #31170514290](https://github.com/sgl-project/sglang/actions/runs/31170514290)
- **分支**: `main_8.5`
- **总耗时**: 12.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31170514290

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147159) |
| stage-b-test-1-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147169) |
| stage-a-unit-test-npu | 6.6min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147172) |
| stage-b-test-4-npu-a3 (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147179) |
| multimodal-gen-test-2-npu-a3 | 7.2min | 环境问题 | GitHub Actions 下载 actions/checkout 和 upload-artifact 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147187) |
| stage-b-test-8-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147199) |
| stage-b-test-4-npu-a3 (1) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147207) |
| stage-b-test-16-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147232) |
| multimodal-gen-test-1-npu-a3 | 6.7min | 环境问题 | GitHub Actions 下载 action 超时，导致作业准备失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147252) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.6min | 其他 | 作业日志不完整，缺少关键失败信息，无法确定具体原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147546) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147592) |

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147159

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147169

- **stage-a-unit-test-npu**: 日志显示在安装Python依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或缓存服务异常，并非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147172

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147179

- **multimodal-gen-test-2-npu-a3**: 日志显示下载 actions/checkout@v4 和 upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍失败，属于网络或 GitHub 服务问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147187

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147199

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147207

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147232

- **multimodal-gen-test-1-npu-a3**: 下载 actions/checkout 和 upload-artifact 时因 HttpClient 超时（100秒）多次重试失败，最终 prepareJob 未完成，报错 jobPod 未设置，作业无法运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147252

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业启动和清理过程，未见测试执行或失败报错，可能因日志截断或作业在早期阶段被取消，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147546

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的模型或数据文件在 Azure Blob 存储中缺失，可能是文件未上传、路径错误或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31170514290/job/92841147592


## [Run #31169784830](https://github.com/sgl-project/sglang/actions/runs/31169784830)
- **分支**: `codex/mm-qwen-owner-transport`
- **总耗时**: 17.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31169784830

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 12.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869179) |
| stage-b-test-16-npu-a3 | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869180) |
| stage-a-unit-test-npu | 5.7min | 其他 | 日志显示测试全部通过，作业实际成功，无失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869192) |
| stage-b-test-4-npu-a3 (1) | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869207) |
| stage-b-test-2-npu-a3 | 12.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869217) |
| multimodal-gen-test-1-npu-a3 | 13.8min | 环境问题 | Git 拉取仓库时网络或服务端异常导致对象缺失，重试后成功。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869238) |
| multimodal-gen-test-2-npu-a3 | 9.1min | 环境问题 | 作业在准备阶段即被终止，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869245) |
| stage-b-test-1-npu-a3 | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869248) |
| stage-b-test-4-npu-a3 (0) | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869377) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.6min | 环境问题 | 作业在准备阶段因GitHub Actions runner环境问题失败，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869636) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869650) |

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869179

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869180

- **stage-a-unit-test-npu**: 日志中所有NPU单元测试均通过（2/2 passed），作业正常完成，仅有Node 20弃用警告，不影响结果。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869192

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869207

- **stage-b-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869217

- **multimodal-gen-test-1-npu-a3**: 首次 fetch 时远程未发送必要对象，报错 'Could not read 5c78727...'，重试后成功，属于临时性网络或服务端问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869238

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在checkout后立即结束，仅有Node 20弃用警告，无测试执行或错误信息，疑似基础设施或调度问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869245

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869248

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869377

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在checkout后运行自定义k8s脚本时中断，仅有Node.js 20弃用警告，无测试执行或错误信息，疑似runner环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869636

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或数据集）已被删除或路径错误，属于环境配置或资源缺失问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169784830/job/92838869650


## [Run #31169718841](https://github.com/sgl-project/sglang/actions/runs/31169718841)
- **分支**: `codex/mm-cuda-ipc-stream-sync`
- **总耗时**: 16.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31169718841

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (1) | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700670) |
| stage-b-test-16-npu-a3 | 12.2min | 环境问题 | 自托管runner执行自定义容器实现失败，下载过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700671) |
| stage-b-test-4-npu-a3 (0) | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700674) |
| stage-b-test-2-npu-a3 | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700680) |
| multimodal-gen-test-2-npu-a3 | 12.6min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700696) |
| multimodal-gen-test-1-npu-a3 | 13.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700715) |
| stage-b-test-1-npu-a3 | 12.8min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700719) |
| stage-b-test-8-npu-a3 | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700792) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.6min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838701150) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 14.8min | 其他 | 日志不完整，未显示测试执行结果，仅包含作业启动和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838701196) |

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700670

- **stage-b-test-16-npu-a3**: 作业在下载文件时（约5250K处）中断，报错"Executing the custom container implementation failed"，属于runner环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700671

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700674

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700680

- **multimodal-gen-test-2-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700696

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未包含multimodal-gen-test实际执行和失败详情，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700715

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在存储中缺失。可能是上游产物未上传、路径错误或存储被清理，需检查 CI 配置或重跑上游作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700719

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700792

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志被截断，缺少测试执行和失败关键信息，无法判断具体失败原因。可能为测试未运行或日志收集问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838701150

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志在测试运行前即结束，未包含任何测试输出或错误信息，无法判断具体失败原因，可能为日志截断或作业被外部中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838701196

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169718841/job/92838700712) |


## [Run #31169283713](https://github.com/sgl-project/sglang/actions/runs/31169283713)
- **分支**: `k3-mtp-extend-kernel`
- **总耗时**: 65.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31169283713

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 9.2min | 代码错误 | NPU PD分离测试用例执行失败，测试脚本返回非零退出码。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331273) |
| multimodal-gen-test-2-npu-a3 | 23.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331280) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 19.1min | 环境问题 | Azure Blob 存储中指定的模型权重文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837332268) |

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py 测试在运行330秒后报错，0/6测试通过，具体错误信息被截断，需查看完整日志定位具体断言或异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331273

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331280

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明模型权重（glm5_top64_pruned_bf16）在存储中缺失或路径错误，可能是上传失败或配置错误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837332268

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 58.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331252) |
| stage-b-test-4-npu-a3 (1) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331281) |
| stage-b-test-1-npu-a3 | 25.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331288) |
| stage-b-test-4-npu-a3 (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331330) |
| stage-b-test-2-npu-a3 | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331345) |
| stage-b-test-8-npu-a3 | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331364) |
| stage-a-unit-test-npu | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837331385) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31169283713/job/92837332154) |


## [Run #31168937205](https://github.com/sgl-project/sglang/actions/runs/31168937205)
- **分支**: `codex/minimax-h3-cross-node-ring`
- **总耗时**: 50.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31168937205

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 23.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31168937205/job/92836180974) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31168937205/job/92836180974

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31168937205/job/92836180906) |


## [Run #31168210322](https://github.com/sgl-project/sglang/actions/runs/31168210322)
- **分支**: `main`
- **总耗时**: 60.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31168210322

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 23.9min | 环境问题 | 作业因未找到diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31168210322/job/92833944354) |
| multimodal-gen-test-1-npu-a3 | 41.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31168210322/job/92833944406) |

- **multimodal-gen-test-2-npu-a3**: 日志显示upload-artifact步骤未找到diffusion-failures/目录，说明测试未生成失败样本，可能因环境配置或前置步骤失败导致测试未运行，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31168210322/job/92833944354

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行或失败的具体错误，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31168210322/job/92833944406


## [Run #31167747653](https://github.com/sgl-project/sglang/actions/runs/31167747653)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 75.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31167747653

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530562) |
| stage-b-test-8-npu-a3 | 28.0min | 环境问题 | Kubernetes Pod 未找到，作业因等待 Pod 超时被终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530604) |
| stage-b-test-2-npu-a3 | 5.8min | 环境问题 | NPU测试脚本因内存不足被系统OOM杀死（exit code 137）。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530613) |
| stage-b-test-16-npu-a3 | 42.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530626) |
| stage-b-test-1-npu-a3 | 6.0min | 环境问题 | NPU测试作业因进程被OOM Killer终止（exit code 137）而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530631) |
| stage-b-test-4-npu-a3 (0) | 1.4min | 环境问题 | NPU CI 作业在安装系统依赖包时被 OOM 杀死（exit code 137）。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530638) |
| stage-b-test-4-npu-a3 (1) | 6.0min | 环境问题 | NPU作业在加载模型权重时进程被OOM杀死（exit code 137）。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530642) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 8.3min | 环境问题 | Kubernetes Pod 未找到，runner 收到关闭信号导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832531124) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen-test-2-npu-a3作业的实际测试命令或错误输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530562

- **stage-b-test-8-npu-a3**: 日志显示 runner 在等待名为 linux-aarch64-a3-8-cn12-001-cgvfn-runner-jwcsp-workflow 的 Pod 时持续收到 404 错误，Pod 不存在或未创建成功，最终因等待超时（exit code 130）导致作业失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530604

- **stage-b-test-2-npu-a3**: 作业在加载模型权重后，进程退出码为137，表明被系统OOM Killer终止。日志显示可用内存约60GB，但加载4个分片后内存耗尽，属于环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530613

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中不存在。可能是文件被误删、路径错误或上传未完成，导致 CI 作业在下载依赖或数据时失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530626

- **stage-b-test-1-npu-a3**: 日志显示在加载模型权重时可用内存仅60.77GB，随后进程被系统以137退出码杀死，表明内存不足导致OOM。这是NPU环境资源限制问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530631

- **stage-b-test-4-npu-a3 (0)**: 日志显示在 apt 安装 libxcb 等 arm64 包时进程被系统 OOM Killer 终止（exit code 137），属于自托管 runner 环境资源不足或容器内存限制导致，非代码问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530638

- **stage-b-test-4-npu-a3 (1)**: 日志显示模型权重加载阶段（Multi-thread loading shards）进程退出码137，通常表示内存不足（OOM）被系统杀死。可用内存60.77GB，但加载7个分片时内存耗尽，属于环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530642

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示等待 Pod 时持续返回 404 NotFound，最终 runner 收到 shutdown 信号退出（exit code 130），属于基础设施/环境问题，非测试代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832531124

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530671) |
| multimodal-gen-test-1-npu-a3 | 50.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832530694) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31167747653/job/92832531179) |


## [Run #31167484212](https://github.com/sgl-project/sglang/actions/runs/31167484212)
- **分支**: `cheng/gc-final-closeout`
- **总耗时**: 69.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31167484212

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 29.4min | 环境问题 | Kubernetes Pod 未找到，作业因基础设施故障失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624644) |
| stage-b-test-1-npu-a3 | 2.9min | 环境问题 | 安装依赖时进程被OOM杀死（exit code 137） | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624653) |
| stage-b-test-2-npu-a3 | 3.0min | 环境问题 | pip安装依赖时进程被kill（exit code 137），疑似内存不足 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624674) |
| multimodal-gen-test-2-npu-a3 | 27.9min | 环境问题 | 作业因未找到diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624677) |
| stage-b-test-16-npu-a3 | 10.6min | 环境问题 | 自托管runner容器异常退出导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624693) |
| stage-b-test-4-npu-a3 (1) | 2.8min | 环境问题 | 安装torch-npu时进程被kill（exit code 137），疑似内存不足或OOM。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624694) |
| stage-b-test-4-npu-a3 (0) | 2.7min | 环境问题 | 安装Rust时进程被OOM杀死（exit code 137） | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624708) |
| multimodal-gen-test-1-npu-a3 | 41.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624742) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.9min | 其他 | 日志被截断，未显示测试执行结果，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831625230) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 12.6min | 环境问题 | Kubernetes Pod 未找到导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831625253) |

- **stage-b-test-8-npu-a3**: 日志显示 runner 在等待 Pod 'linux-aarch64-a3-8-cn12-001-cgvfn-runner-hk85q-workflow' 时持续收到 404 错误，最终 runner 收到关闭信号退出，属于自托管 runner 基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624644

- **stage-b-test-1-npu-a3**: 在pip安装triton-ascend及其依赖时，卸载旧版numpy/scipy等包过程中内存不足，导致脚本被系统kill（exit code 137），作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624653

- **stage-b-test-2-npu-a3**: 在安装triton-ascend依赖时，卸载numpy过程中进程被系统OOM killer终止（exit code 137），导致安装失败。这可能是由于NPU环境内存资源不足或并发安装导致的内存峰值过高。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624674

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures工件时提示无文件，说明测试未产生失败样本，作业可能因前置条件未满足或测试未运行而终止，属于环境或流程配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624677

- **stage-b-test-16-npu-a3**: 作业在加载模型权重时容器突然消失，报错container not found，属于runner基础设施问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624693

- **stage-b-test-4-npu-a3 (1)**: 在安装torch-npu==2.10.0时，脚本以exit code 137退出，通常表示进程被系统OOM killer终止，可能因内存不足或资源限制导致。建议检查runner内存配置或减少并发安装任务。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624694

- **stage-b-test-4-npu-a3 (0)**: 在安装Rust 1.92过程中，下载rustc组件时内存不足，进程被系统OOM killer终止（exit code 137），导致作业失败。属于NPU环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624708

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions的初始化、上传artifact（无文件）和清理步骤，未包含multimodal测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624742

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业初始化和清理阶段，未展示测试运行过程及错误信息，需查看完整日志以定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831625230

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 作业等待的 Pod `linux-aarch64-a3-16-cn12-001-69mtf-runner-bjglg-workflow` 返回 404 不存在，多次重试后 runner 收到关闭信号，退出码 130，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831625253

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31167484212/job/92831624646) |


## [Run #31167376533](https://github.com/sgl-project/sglang/actions/runs/31167376533)
- **分支**: `marv/ar_norm_per_token_quant_fusion`
- **总耗时**: 70.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31167376533

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 2.0min | 环境问题 | 安装依赖时进程被系统杀死（exit code 137），疑似内存不足。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309752) |
| stage-b-test-4-npu-a3 (1) | 4.3min | 环境问题 | pip安装依赖时进程被OOM杀死（exit code 137） | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309770) |
| multimodal-gen-test-2-npu-a3 | 26.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309773) |
| stage-b-test-1-npu-a3 | 4.3min | 环境问题 | pip安装依赖时进程被OOM杀死（exit code 137） | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309786) |
| stage-b-test-8-npu-a3 | 30.3min | 环境问题 | Kubernetes Pod 未找到导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309787) |
| stage-b-test-4-npu-a3 (0) | 36.8min | 环境问题 | Kubernetes Pod 未找到，作业因等待 Pod 超时被终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309790) |
| multimodal-gen-test-1-npu-a3 | 44.6min | 其他 | 作业日志被截断，未显示实际测试结果，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309810) |
| stage-b-test-16-npu-a3 | 39.1min | 环境问题 | Kubernetes Pod 未找到导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309827) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 33.4min | 环境问题 | Kubernetes Pod 不存在导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831310173) |

- **stage-b-test-2-npu-a3**: 在下载并安装torch等大型包时，脚本以退出码137终止，通常表示OOM（内存不足）被kill。日志显示下载未完成即失败，属于NPU runner环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309752

- **stage-b-test-4-npu-a3 (1)**: 在安装triton-ascend及其依赖（numpy、pytest等）时，进程因内存不足被系统杀死（exit code 137），导致安装中断，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309770

- **multimodal-gen-test-2-npu-a3**: 日志显示作业启动后执行了上传artifact步骤，但未找到diffusion-failures目录，随后进入清理阶段。未看到任何测试执行或失败断言，可能因日志截断或作业提前结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309773

- **stage-b-test-1-npu-a3**: 在安装triton-ascend及其依赖（如triton 159.9MB、pandas等）时，runner内存不足导致进程被系统kill（exit code 137），属于NPU CI环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309786

- **stage-b-test-8-npu-a3**: 作业启动后，等待的 Pod（linux-aarch64-a3-8-cn12-001-cgvfn-runner-7mhwc-workflow）始终返回404，持续重试约30分钟后被取消，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309787

- **stage-b-test-4-npu-a3 (0)**: 日志显示 runner 在等待名为 linux-aarch64-a3-4-cn12-001-hcf8l-runner-26r5d-workflow 的 Pod 时持续收到 404 错误，Pod 不存在，最终因超时（exit code 130）被取消，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309790

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败原因。仅能看到上传diffusion-failures目录时提示无文件，可能测试未产生失败样本或测试未执行。需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309810

- **stage-b-test-16-npu-a3**: 作业启动后，等待的 Kubernetes Pod（linux-aarch64-a3-16-cn12-001-69mtf-runner-z8fqc-workflow）始终返回 404 未找到，重试约 40 分钟后 runner 收到关闭信号退出，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309827

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 作业等待的 Pod (linux-aarch64-a3-16-cn12-001-69mtf-runner-7bf68-workflow) 返回404 Not Found，多次重试后 runner 收到关闭信号退出，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831310173

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831309816) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31167376533/job/92831310322) |


## [Run #31166702805](https://github.com/sgl-project/sglang/actions/runs/31166702805)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 14.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31166702805

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 0.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255833) |
| stage-b-test-16-npu-a3 | 3.2min | 环境问题 | Kubernetes Pod 未找到导致作业启动失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255841) |
| stage-b-test-1-npu-a3 | 1.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255842) |
| multimodal-gen-test-1-npu-a3 | 12.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255856) |
| stage-b-test-4-npu-a3 (0) | 8.1min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255860) |
| multimodal-gen-test-2-npu-a3 | 12.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255885) |
| stage-b-test-4-npu-a3 (1) | 8.1min | 环境问题 | 下载triton-ascend依赖时自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255908) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 3.3min | 环境问题 | Kubernetes Pod 未找到导致作业启动失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829256078) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 12.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829256094) |

- **stage-b-test-2-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于基础设施或容器配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255833

- **stage-b-test-16-npu-a3**: Runner 尝试读取 workflow Pod 时持续返回 404 NotFound，重试后仍失败，最终自定义容器执行失败。属于基础设施环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255841

- **stage-b-test-1-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于runner或容器环境配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255842

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问某个blob时，返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置问题，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255856

- **stage-b-test-4-npu-a3 (0)**: 日志显示在下载依赖过程中，runner执行自定义容器实现时出错，提示联系管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255860

- **multimodal-gen-test-2-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件可能已被删除、路径错误或未上传成功，属于外部依赖资源缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255885

- **stage-b-test-4-npu-a3 (1)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业中断，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255908

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: runner 在准备阶段无法找到对应的 workflow Pod（HTTP 404），导致 prepareJob 未完成，后续步骤因 jobPod 未设置而失败。属于自托管 runner 基础设施问题，非测试代码或模型精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829256078

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误信息为BlobNotFound，说明作业尝试访问的Azure Blob存储资源缺失或路径错误，可能是CI配置中引用的文件未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829256094

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31166702805/job/92829255830) |


## [Run #31166107840](https://github.com/sgl-project/sglang/actions/runs/31166107840)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 5.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31166107840

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268551) |
| stage-b-test-16-npu-a3 | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268594) |
| stage-b-test-4-npu-a3 (0) | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268626) |
| stage-a-unit-test-npu | 4.9min | 环境问题 | 自定义容器执行失败，测试本身全部通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268638) |
| stage-b-test-2-npu-a3 | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268652) |
| stage-b-test-4-npu-a3 (1) | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268669) |
| multimodal-gen-test-1-npu-a3 | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268682) |
| stage-b-test-8-npu-a3 | 5.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268711) |
| multimodal-gen-test-2-npu-a3 | 5.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268723) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827269159) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827269170) |

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268551

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或文件被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268594

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268626

- **stage-a-unit-test-npu**: NPU单元测试（80+50个用例）全部通过，但作业在测试结束后因自定义容器实现执行失败而报错，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268638

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268652

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268669

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268682

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268711

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问某个blob时，返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置问题，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827268723

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是配置问题或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827269159

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31166107840/job/92827269170


## [Run #31165526007](https://github.com/sgl-project/sglang/actions/runs/31165526007)
- **分支**: `codex/kimi-k3-npu-main-20260803`
- **总耗时**: 86.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31165526007

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 5.9min | 环境问题 | NPU HiCache MHA测试失败，服务启动异常导致测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331662) |
| stage-b-test-2-npu-a3 | 5.6min | 环境问题 | NPU测试用例启动sglang服务失败，导致测试全部失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331732) |
| stage-b-test-4-npu-a3 (1) | 9.2min | 环境问题 | 构建xatlas依赖时进程被OOM killer杀死（exit code 137），导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331740) |
| stage-b-test-16-npu-a3 | 13.8min | 环境问题 | Kubernetes Pod 未找到导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331764) |
| stage-b-test-8-npu-a3 | 11.2min | 环境问题 | Kubernetes Pod 未找到导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331798) |
| multimodal-gen-test-2-npu-a3 | 8.3min | 其他 | 作业未执行实际测试，仅上传空失败目录后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331804) |
| stage-b-test-4-npu-a3 (0) | 8.7min | 环境问题 | 构建过程中进程被系统杀死（exit code 137），疑似内存不足或资源限制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331911) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.9min | 其他 | 作业在启动阶段即被终止，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827332773) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 13.9min | 环境问题 | Kubernetes Pod 未找到，作业因基础设施故障失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827333060) |

- **stage-b-test-1-npu-a3**: 测试test_npu_hicache_mha.py启动sglang服务失败，命令包含--enable-hierarchical-cache和--hicache-ratio 1.2，可能因NPU环境或配置问题导致服务无法正常运行，测试0/11通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331662

- **stage-b-test-2-npu-a3**: 测试test_npu_mla_fia_w8a8int8.py启动sglang serve命令失败（exit code 255），服务未能正常启动，导致0/6测试全部失败。可能是NPU环境配置或模型加载问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331732

- **stage-b-test-4-npu-a3 (1)**: 在安装xatlas 0.0.11包时，CMake/Ninja编译过程内存不足，进程被系统以137码终止，属于NPU环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331740

- **stage-b-test-16-npu-a3**: 作业尝试读取名为 linux-aarch64-a3-16-cn12-001-69mtf-runner-pt7gq-workflow 的 Pod 时持续返回 404 NotFound，重试多次后 runner 收到关闭信号退出，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331764

- **stage-b-test-8-npu-a3**: 作业启动后，等待的 workflow Pod 持续返回 404 NotFound，多次重试后 runner 收到关闭信号退出，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331798

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未发现测试运行记录，且diffusion-failures目录无文件，作业可能因前置条件未满足而跳过测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331804

- **stage-b-test-4-npu-a3 (0)**: 在运行setuptools构建sglang包时，进程因OOM被kill（exit code 137），导致脚本失败。日志显示构建过程正常，但突然被终止，属于环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331911

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试用例，可能因基础设施问题或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827332773

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示等待 Pod 时持续返回 404 Not Found，最终 runner 收到关闭信号退出（exit code 130），属于自托管 runner 环境问题，非测试代码或模型精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827333060

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331696) |
| multimodal-gen-test-1-npu-a3 | 45.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31165526007/job/92827331899) |


## [Run #31164898084](https://github.com/sgl-project/sglang/actions/runs/31164898084)
- **分支**: `add-inkling-cache-consistency-test`
- **总耗时**: 91.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31164898084

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465611) |
| stage-b-test-8-npu-a3 | 11.3min | 环境问题 | Kubernetes Pod 未找到导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465788) |
| stage-b-test-16-npu-a3 | 48.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465840) |
| stage-b-test-4-npu-a3 (0) | 11.4min | 环境问题 | NPU测试进程被系统OOM杀死，退出码137 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465861) |
| stage-b-test-2-npu-a3 | 12.2min | 环境问题 | NPU测试作业因容器内存不足被系统OOM杀死（exit code 137）。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823466071) |
| stage-b-test-4-npu-a3 (1) | 9.4min | 环境问题 | 构建过程中进程被系统杀死（exit code 137），疑似内存不足或资源限制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823466165) |
| stage-b-test-1-npu-a3 | 12.2min | 环境问题 | NPU测试进程被系统OOM杀死（exit code 137），导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823466432) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 48.2min | 环境问题 | Azure Blob 存储中指定的模型文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823466584) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未出现测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465611

- **stage-b-test-8-npu-a3**: 作业尝试读取名为 linux-aarch64-a3-8-cn12-001-cgvfn-runner-9zfpj-workflow 的 Pod 时持续返回 404，多次重试后 runner 收到关闭信号退出，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465788

- **stage-b-test-16-npu-a3**: 作业在下载或访问某个Azure Blob存储资源时，返回BlobNotFound错误（HTTP 404），说明该资源已被删除、路径错误或未上传，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465840

- **stage-b-test-4-npu-a3 (0)**: 测试test_npu_hicache_mla.py启动后立即被kill（exit code 137），表明进程因内存不足被OOM killer终止，属于NPU环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465861

- **stage-b-test-2-npu-a3**: 作业在加载模型权重时可用内存仅约61GB，加载到19%时进程被OOM killer终止，退出码137，导致容器异常退出。属于资源限制导致的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823466071

- **stage-b-test-4-npu-a3 (1)**: 在安装ninja并配置CMake时，进程以137退出（通常为OOM或被kill），可能是NPU节点资源不足或容器内存限制导致构建中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823466165

- **stage-b-test-1-npu-a3**: 在运行第二个测试test_npu_autoround_dense.py时，进程因内存不足被系统kill（exit code 137），可能是NPU显存或主机内存不足，属于环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823466432

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明模型权重或数据文件在存储中缺失或路径错误，可能是上传失败或配置错误，需检查存储路径及文件是否完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823466584

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465660) |
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823465857) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164898084/job/92823467041) |


## [Run #31164787815](https://github.com/sgl-project/sglang/actions/runs/31164787815)
- **分支**: `new_layernorm`
- **总耗时**: 77.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31164787815

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 4.5min | 环境问题 | 安装triton-ascend时进程被OOM杀死（exit code 137）。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036519) |
| stage-b-test-16-npu-a3 | 13.8min | 环境问题 | Kubernetes Pod 未找到导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036532) |
| stage-b-test-1-npu-a3 | 4.4min | 环境问题 | NPU测试脚本执行时被系统OOM杀死（exit code 137），导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036564) |
| stage-b-test-8-npu-a3 | 10.7min | 环境问题 | Kubernetes Pod 未找到，作业因等待 Pod 超时被终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036567) |
| stage-b-test-4-npu-a3 (1) | 4.3min | 环境问题 | 自托管runner在安装系统依赖包时被OOM杀死（exit code 137）。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036578) |
| stage-b-test-4-npu-a3 (0) | 4.1min | 环境问题 | 安装triton-ascend时进程被OOM杀死（exit code 137） | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036582) |
| multimodal-gen-test-2-npu-a3 | 24.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036585) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 13.7min | 环境问题 | Kubernetes Pod 未找到，runner 收到关闭信号导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830037234) |

- **stage-b-test-2-npu-a3**: 在下载triton-ascend 188.5MB的wheel包时，脚本因内存不足被系统kill（exit code 137），导致安装中断，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036519

- **stage-b-test-16-npu-a3**: 作业启动后，等待的 Kubernetes Pod（linux-aarch64-a3-16-cn12-001-69mtf-runner-f47k5-workflow）始终返回 404 未找到，重试多次后 runner 收到关闭信号退出，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036532

- **stage-b-test-1-npu-a3**: 首个测试test_npu_gptq_moe.py启动后立即被kill，退出码137表示内存不足（OOM）。可能是NPU资源不足或测试环境内存配置问题，需检查runner资源分配。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036564

- **stage-b-test-8-npu-a3**: 日志显示 runner 在等待名为 linux-aarch64-a3-8-cn12-001-cgvfn-runner-g2262-workflow 的 Pod 时持续收到 404 错误，Pod 不存在，最终因超时收到关闭信号退出（exit code 130）。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036567

- **stage-b-test-4-npu-a3 (1)**: 日志显示在apt安装libx11等包时进程被kill，退出码137表示内存不足（OOM），属于runner环境资源问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036578

- **stage-b-test-4-npu-a3 (0)**: 在下载triton-ascend 188.5MB的wheel包时，进程因内存不足被系统kill（exit code 137），导致安装失败。这是NPU环境资源限制问题，非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036582

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行和失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或失败原因被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036585

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示等待 Pod 时持续返回 404 NotFound，最终 runner 收到 shutdown 信号退出（exit code 130），属于基础设施/调度环境异常，非测试代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830037234

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036601) |
| multimodal-gen-test-1-npu-a3 | 46.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830036629) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164787815/job/92830037211) |


## [Run #31164395932](https://github.com/sgl-project/sglang/actions/runs/31164395932)
- **分支**: `gptoss-mxfp4`
- **总耗时**: 91.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31164395932

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 11.2min | 环境问题 | Kubernetes Pod 未创建导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884133) |
| stage-b-test-1-npu-a3 | 12.1min | 环境问题 | NPU测试执行过程中进程异常退出，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884145) |
| stage-b-test-4-npu-a3 (1) | 9.5min | 环境问题 | 安装依赖时进程被OOM杀死（exit code 137） | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884181) |
| multimodal-gen-test-2-npu-a3 | 24.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884203) |
| stage-b-test-16-npu-a3 | 39.4min | 环境问题 | Kubernetes Pod 未找到导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884211) |
| stage-b-test-2-npu-a3 | 12.1min | 环境问题 | NPU测试作业在启动服务时失败，进程退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884270) |
| stage-b-test-4-npu-a3 (0) | 9.5min | 环境问题 | 自定义容器执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884318) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 13.9min | 环境问题 | Kubernetes Pod 未找到，runner 收到关闭信号导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884799) |

- **stage-b-test-8-npu-a3**: runner 等待名为 linux-aarch64-a3-8-cn12-001-cgvfn-runner-r9wxr-workflow 的 Pod 时持续收到 404 NotFound 错误，Pod 未能成功创建，最终 runner 收到关闭信号退出（exit code 130）。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884133

- **stage-b-test-1-npu-a3**: 日志显示测试在捕获批次时出现大量torch_npu警告，随后进程以退出码1终止，可能是NPU环境不稳定或资源问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884145

- **stage-b-test-4-npu-a3 (1)**: 在pip安装triton-ascend等依赖时，卸载旧包过程中内存不足，导致脚本被系统kill（exit code 137），属于NPU环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884181

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js版本弃用警告和上传artifact时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884203

- **stage-b-test-16-npu-a3**: Runner 在等待创建 workflow Pod 时持续收到 404 错误，Pod 始终未创建成功，最终因超时收到关闭信号退出（exit code 130）。属于基础设施/调度问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884211

- **stage-b-test-2-npu-a3**: 日志显示服务启动参数正常，但在初始化过程中进程异常退出（exit code 1），可能是NPU资源分配失败或环境配置问题，需检查自托管runner的NPU环境。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884270

- **stage-b-test-4-npu-a3 (0)**: 作业在运行自定义容器时失败，日志显示依赖安装成功但随后进程退出码1，可能是容器环境或资源问题导致，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884318

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示等待 Pod 时持续返回 404 NotFound，最终 runner 收到 shutdown 信号退出（exit code 130），属于基础设施/环境问题，非测试代码或模型精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884799

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884094) |
| multimodal-gen-test-1-npu-a3 | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884123) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31164395932/job/92821884716) |


---
*Auto-generated by npu_pr_monitor.py*