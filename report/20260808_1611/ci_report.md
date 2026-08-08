# NPU CI 执行监控
**生成时间**: 2026-08-08 08:11 UTC
**分析 Run 数**: 10

---

## [Run #31245652923](https://github.com/sgl-project/sglang/actions/runs/31245652923)
- **分支**: `codex/diffusion-kernel-cleanup`
- **总耗时**: 43.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31245652923

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 25.9min | 代码错误 | NPU PD分离测试用例失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757919) |
| multimodal-gen-test-2-npu-a3 | 22.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757973) |

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py测试失败，2/6用例通过，该用例耗时314秒后退出码为1，可能涉及PD分离功能逻辑错误或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757919

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有GitHub Actions环境准备、Node版本警告及上传失败产物（无文件）等常规信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757973

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757909) |
| stage-b-test-8-npu-a3 | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757917) |
| stage-b-test-2-npu-a3 | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757918) |
| stage-b-test-1-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757920) |
| stage-b-test-4-npu-a3 (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757938) |
| stage-b-test-4-npu-a3 (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757954) |
| multimodal-gen-test-1-npu-a3 | 26.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073757976) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073758121) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245652923/job/93073758137) |


## [Run #31245192576](https://github.com/sgl-project/sglang/actions/runs/31245192576)
- **分支**: `main`
- **总耗时**: 54.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31245192576

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 9.6min | 代码错误 | NPU HiCache MHA 测试失败，测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607565) |
| stage-b-test-16-npu-a3 | 38.5min | 代码错误 | NPU PD分离测试用例失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607573) |
| stage-b-test-8-npu-a3 | 13.8min | 代码错误 | NPU EP LB 最小再平衡利用率阈值测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607574) |
| stage-b-test-4-npu-a3 (0) | 7.8min | 代码错误 | HiCache MLA 测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607579) |
| multimodal-gen-test-2-npu-a3 | 22.5min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607605) |

- **stage-b-test-1-npu-a3**: 测试 test_npu_hicache_mha.py 执行失败（exit code 1），导致整体测试通过率仅2/11。该测试涉及HiCache功能，可能是代码逻辑或环境配置问题，需进一步查看具体错误输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607565

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py测试失败，6个测试中2个通过，该用例执行349秒后返回退出码1，具体断言或运行错误需查看详细日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607573

- **stage-b-test-8-npu-a3**: 测试文件 test_npu_eplb_min_rebalancing_utilization_threshold.py 执行失败，返回退出码1，耗时648秒超过预估400秒，但未显示具体断言错误，可能涉及配置或逻辑问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607574

- **stage-b-test-4-npu-a3 (0)**: test_npu_hicache_mla.py 测试运行 267 秒后报错，返回退出码 1，导致整个测试阶段 0/5 通过。具体错误信息未在日志中显示，但可判断为测试用例本身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607579

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，未运行实际测试或出现明确错误，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607605

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607552) |
| stage-b-test-4-npu-a3 (1) | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607582) |
| stage-b-test-2-npu-a3 | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607598) |
| multimodal-gen-test-1-npu-a3 | 27.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607612) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607805) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31245192576/job/93072607817) |


## [Run #31245085082](https://github.com/sgl-project/sglang/actions/runs/31245085082)
- **分支**: `codex/diffusion-kernel-cleanup`
- **总耗时**: 9.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31245085082

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882849) |
| stage-b-test-4-npu-a3 (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882850) |
| stage-b-test-2-npu-a3 | 3.1min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882866) |
| stage-b-test-16-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882870) |
| stage-a-unit-test-npu | 3.9min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882880) |
| multimodal-gen-test-2-npu-a3 | 5.5min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882881) |
| stage-b-test-4-npu-a3 (1) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882884) |
| multimodal-gen-test-1-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882885) |
| stage-b-test-1-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882891) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072883144) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 8.6min | 环境问题 | 依赖的Blob资源不存在导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072883193) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882849

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882850

- **stage-b-test-2-npu-a3**: test/registered/kernels/ops/diffusion/test_quality_gate.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 收集测试时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882866

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882870

- **stage-a-unit-test-npu**: test/registered/kernels/ops/diffusion/test_quality_gate.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在直接运行时被静默跳过，CI 收集测试时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882880

- **multimodal-gen-test-2-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882881

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882884

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882885

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072882891

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072883144

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示Azure Blob存储返回BlobNotFound错误，说明作业所需的模型权重或数据文件未上传或已被删除，属于环境/资源准备问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31245085082/job/93072883193


## [Run #31244407894](https://github.com/sgl-project/sglang/actions/runs/31244407894)
- **分支**: `main`
- **总耗时**: 12.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31244407894

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 9.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555912) |
| stage-b-test-2-npu-a3 | 9.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555916) |
| stage-b-test-16-npu-a3 | 4.2min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555921) |
| multimodal-gen-test-1-npu-a3 | 11.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555974) |
| stage-b-test-4-npu-a3 (1) | 11.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555975) |
| stage-b-test-4-npu-a3 (0) | 7.8min | 环境问题 | NPU HiCache MLA测试失败，服务启动后测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555984) |
| stage-b-test-1-npu-a3 | 9.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555985) |
| multimodal-gen-test-2-npu-a3 | 11.3min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志被截断无法定位真实失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555992) |
| stage-a-unit-test-npu | 1.1min | 环境问题 | NPU CI 作业因容器镜像拉取失败（ImagePullBackOff）而无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555994) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 11.5min | 其他 | 日志不完整，未显示测试执行结果，仅包含作业启动和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070556235) |

- **stage-b-test-8-npu-a3**: 作业在运行测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555912

- **stage-b-test-2-npu-a3**: 作业在TokenizerManager初始化后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555916

- **stage-b-test-16-npu-a3**: 日志显示在安装Rust组件时，runner报错“Executing the custom container implementation failed”，提示联系管理员，属于runner环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555921

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js 20弃用警告和上传artifact时未找到diffusion-failures目录的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555974

- **stage-b-test-4-npu-a3 (1)**: 日志显示测试运行正常，但在06:51:13时出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555975

- **stage-b-test-4-npu-a3 (0)**: 测试test_npu_hicache_mla.py在281秒后失败，0/5测试通过。服务启动命令使用DeepSeek-V2-Lite-W8A8模型和HiCache配置，但测试未通过，可能是环境配置或模型兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555984

- **stage-b-test-1-npu-a3**: 日志显示测试运行正常（进度94%），但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555985

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但提示无文件可上传，且中间关键测试日志被省略，无法判断具体失败点，可能为测试未运行或结果未生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555992

- **stage-a-unit-test-npu**: K8s 无法从华为云镜像仓库拉取 cann:9.0.0-910b-ubuntu22.04-py3.11 镜像，可能是镜像不存在、凭据无效或网络不通，导致 pod 一直处于 Pending 状态，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070555994

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 提供的日志片段仅包含GitHub Actions运行器初始化、作业输入配置及结束时的清理步骤，未包含实际测试命令执行、错误堆栈或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070556235

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244407894/job/93070556170) |


## [Run #31244185437](https://github.com/sgl-project/sglang/actions/runs/31244185437)
- **分支**: `optimize-mem-pool-slot-alloc`
- **总耗时**: 31.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31244185437

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 9.0min | 环境问题 | NPU PD分离测试失败，测试用例执行报错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019112) |
| multimodal-gen-test-2-npu-a3 | 25.8min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019122) |

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py测试在323秒后失败，返回错误码1，0/6测试通过。可能是NPU环境配置问题或测试用例本身存在缺陷，需进一步查看详细错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019112

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node版本弃用警告和上传artifact时无文件提示，无法判断具体失败点，可能为作业被中断或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019122

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019100) |
| stage-b-test-4-npu-a3 (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019108) |
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019109) |
| stage-b-test-4-npu-a3 (1) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019114) |
| stage-b-test-8-npu-a3 | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019120) |
| stage-b-test-1-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019127) |
| stage-b-test-2-npu-a3 | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019143) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019395) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244185437/job/93070019401) |


## [Run #31244083469](https://github.com/sgl-project/sglang/actions/runs/31244083469)
- **分支**: `inkling-tool-result-multimodal`
- **总耗时**: 9.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31244083469

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 6.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733213) |
| stage-b-test-8-npu-a3 | 6.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733219) |
| stage-b-test-2-npu-a3 | 6.3min | 环境问题 | 自定义容器执行失败，NPU后端不支持CUDA相关操作导致服务异常退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733222) |
| stage-b-test-16-npu-a3 | 6.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733231) |
| multimodal-gen-test-1-npu-a3 | 6.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733248) |
| stage-b-test-4-npu-a3 (1) | 6.6min | 环境问题 | 自定义容器执行失败，NPU初始化过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733250) |
| stage-b-test-4-npu-a3 (0) | 6.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733252) |
| multimodal-gen-test-2-npu-a3 | 6.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733260) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 8.2min | 环境问题 | 作业在启动后立即被清理，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733468) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.7min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733469) |

- **stage-b-test-1-npu-a3**: 作业在运行第二个测试时，容器实现执行失败，日志显示'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733213

- **stage-b-test-8-npu-a3**: 作业在启动模型加载阶段（TP3 EP3）后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733219

- **stage-b-test-2-npu-a3**: 日志显示SymmetricMemory不支持cuda设备类型，且NPU后端对aten::_assert_async算子回退到CPU执行，最终自定义容器实现执行失败，属于环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733222

- **stage-b-test-16-npu-a3**: 作业在register_tokenizer步骤成功后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733231

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有GitHub Actions运行器初始化、Node版本警告及上传artifact时未找到文件的提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733248

- **stage-b-test-4-npu-a3 (1)**: 作业在torch分布式初始化阶段（Init torch distributed begin）后立即报错，提示自定义容器实现执行失败，可能是NPU环境配置或容器启动问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733250

- **stage-b-test-4-npu-a3 (0)**: 日志显示在测试运行过程中，自定义容器实现执行失败，提示联系自托管runner管理员。可能是容器资源限制或环境配置问题导致，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733252

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733260

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试命令，可能是runner环境异常或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733468

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含作业初始化、环境准备和清理步骤，未展示测试运行过程及失败点，可能因日志截断或作业在测试前被中断，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733469

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31244083469/job/93069733214) |


## [Run #31243936499](https://github.com/sgl-project/sglang/actions/runs/31243936499)
- **分支**: `optimize-mem-pool-slot-alloc`
- **总耗时**: 7.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31243936499

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343478) |
| stage-b-test-1-npu-a3 | 2.2min | 环境问题 | 自定义容器执行失败，导致作业在构建阶段中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343480) |
| stage-b-test-2-npu-a3 | 1.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343495) |
| stage-b-test-16-npu-a3 | 1.8min | 环境问题 | 自定义容器执行失败，导致作业在安装依赖后中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343500) |
| multimodal-gen-test-1-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343505) |
| stage-b-test-4-npu-a3 (1) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343506) |
| multimodal-gen-test-2-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343508) |
| stage-b-test-4-npu-a3 (0) | 1.9min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343510) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.0min | 其他 | 日志被截断，未显示实际测试结果，无法确定失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343658) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343672) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343478

- **stage-b-test-1-npu-a3**: 日志显示在运行自定义容器实现时出错（Executing the custom container implementation failed），可能是NPU A3环境配置或容器启动问题，并非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343480

- **stage-b-test-2-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343495

- **stage-b-test-16-npu-a3**: 日志显示在安装torch等依赖后，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343500

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343505

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径和生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343506

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的模型/数据文件未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343508

- **stage-b-test-4-npu-a3 (0)**: 日志显示在安装triton-ascend等依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343510

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 提供的日志仅包含作业初始化和清理阶段，未包含测试执行及失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343658

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343672

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243936499/job/93069343492) |


## [Run #31243716958](https://github.com/sgl-project/sglang/actions/runs/31243716958)
- **分支**: `fix-dspark-cp`
- **总耗时**: 36.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31243716958

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 10.0min | 超时 | NPU PD分离测试超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813931) |
| multimodal-gen-test-2-npu-a3 | 22.4min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813933) |
| stage-b-test-4-npu-a3 (0) | 14.6min | 代码错误 | NPU DP注意力测试失败，测试用例返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813937) |

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py 测试运行401秒，超过预估的400秒限制，导致测试失败，作业整体退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813931

- **multimodal-gen-test-2-npu-a3**: 日志中未包含任何测试执行或失败信息，仅有GitHub Actions运行器初始化、Node版本警告及上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813933

- **stage-b-test-4-npu-a3 (0)**: test_npu_dp_attention.py测试失败，耗时471秒超过预估400秒，可能因超时或断言失败导致退出码1，需检查该测试用例的具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813937

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813920) |
| stage-b-test-1-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813940) |
| stage-b-test-8-npu-a3 | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813949) |
| stage-b-test-4-npu-a3 (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813953) |
| stage-b-test-2-npu-a3 | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813965) |
| multimodal-gen-test-1-npu-a3 | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068813966) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068814362) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31243716958/job/93068814382) |


## [Run #31242910500](https://github.com/sgl-project/sglang/actions/runs/31242910500)
- **分支**: `main`
- **总耗时**: 39.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31242910500

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 36.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771759) |
| stage-b-test-4-npu-a3 (0) | 8.7min | 代码错误 | NPU HiCache MLA 测试失败，测试用例执行报错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771770) |
| multimodal-gen-test-2-npu-a3 | 8.4min | 其他 | 作业未执行实际测试，仅上传空artifact后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771786) |
| stage-b-test-1-npu-a3 | 37.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771788) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.9min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771892) |

- **stage-b-test-16-npu-a3**: 作业在服务启动后立即报错"Executing the custom container implementation failed"，提示联系self-hosted runner管理员，属于NPU容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771759

- **stage-b-test-4-npu-a3 (0)**: test_npu_hicache_mla.py 测试在 NPU A3 上运行约 282 秒后失败，返回退出码 1，测试摘要显示 0/5 通过。具体错误信息未在日志中详细展示，但可判断为测试用例本身执行出错，可能涉及 HiCache 功能或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771770

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，且未找到diffusion-failures文件，无测试执行记录，可能因前置条件未满足或测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771786

- **stage-b-test-1-npu-a3**: 日志显示测试运行正常，但在06:38:06时出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771788

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业初始化和清理阶段，未展示测试运行输出或错误信息，可能因日志截断或作业在测试前被中断，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771892

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771764) |
| multimodal-gen-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771778) |
| stage-b-test-4-npu-a3 (1) | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771781) |
| stage-b-test-2-npu-a3 | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771782) |
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771795) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242910500/job/93066771896) |


## [Run #31242304715](https://github.com/sgl-project/sglang/actions/runs/31242304715)
- **分支**: `main`
- **总耗时**: 16.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31242304715

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 9.2min | 代码错误 | NPU PD disaggregation 测试失败，测试用例执行报错。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257215) |
| multimodal-gen-test-1-npu-a3 | 16.0min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257218) |
| multimodal-gen-test-2-npu-a3 | 14.1min | 其他 | 作业未执行实际测试，仅上传空失败产物后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257224) |
| stage-b-test-2-npu-a3 | 14.3min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257239) |
| stage-b-test-4-npu-a3 (1) | 12.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257243) |
| stage-b-test-1-npu-a3 | 15.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257248) |
| stage-b-test-4-npu-a3 (0) | 7.7min | 精度回归 | HiCache MLA 测试精度失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257352) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 16.1min | 环境问题 | 作业在准备阶段被中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257435) |

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py 测试运行 327 秒后失败，返回 exit code 1，测试摘要显示 0/6 通过。具体错误信息未在日志中显示，但可判断为测试用例本身执行出错，可能涉及 PD disaggregation 功能逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257215

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行或失败的具体内容，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257218

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，随后正常清理退出。未出现测试执行、错误或超时信息，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257224

- **stage-b-test-2-npu-a3**: 测试运行到第2个用例时，自定义容器实现执行失败，导致作业终止。日志显示测试本身通过，但容器环境异常中断了后续测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257239

- **stage-b-test-4-npu-a3 (1)**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257243

- **stage-b-test-1-npu-a3**: 日志显示测试运行正常（进度55%），但突然报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257248

- **stage-b-test-4-npu-a3 (0)**: test_npu_hicache_mla.py 在 DeepSeek-V2-Lite-W8A8 模型上精度测试未通过，0/4 用例通过，耗时282秒，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257352

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在备份plog文件后进入清理阶段，未执行测试用例，可能因runner环境异常或作业被提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257435

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257232) |
| stage-b-test-8-npu-a3 | 11.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257249) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31242304715/job/93065257462) |


---
*Auto-generated by npu_pr_monitor.py*