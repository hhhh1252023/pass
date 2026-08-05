# NPU CI 执行监控
**生成时间**: 2026-08-05 12:12 UTC
**分析 Run 数**: 19

---

## [Run #31000091356](https://github.com/sgl-project/sglang/actions/runs/31000091356)
- **分支**: `main`
- **总耗时**: 9.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31000091356

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31000091356/job/92286727025) |
| multimodal-gen-test-2-npu-a3 | 9.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31000091356/job/92286727086) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31000091356/job/92286727025

- **multimodal-gen-test-2-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31000091356/job/92286727086


## [Run #30999655422](https://github.com/sgl-project/sglang/actions/runs/30999655422)
- **分支**: `main`
- **总耗时**: 6.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30999655422

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30999655422/job/92285305191) |
| multimodal-gen-test-2-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30999655422/job/92285305294) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的模型/数据文件未上传或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30999655422/job/92285305191

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源未上传或已被删除，属于环境配置或资源缺失问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30999655422/job/92285305294


## [Run #30992447131](https://github.com/sgl-project/sglang/actions/runs/30992447131)
- **分支**: `mmangkad/torch-2.12`
- **总耗时**: 154.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30992447131

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 2.5min | 环境问题 | 下载依赖包时代理返回HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723487) |
| stage-b-test-1-npu-a3 | 2.3min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723522) |
| stage-b-test-8-npu-a3 | 2.4min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723532) |
| stage-b-test-16-npu-a3 | 2.5min | 环境问题 | 下载sgl-kernel-npu依赖时HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723565) |
| multimodal-gen-test-1-npu-a3 | 15.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723571) |
| multimodal-gen-test-2-npu-a3 | 16.2min | 其他 | 作业未发现明确失败原因，日志显示正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723576) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.8min | 环境问题 | 作业在启动阶段即失败，未进入实际测试，日志显示缺少关键执行步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261724168) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.7min | 环境问题 | 作业在准备阶段被中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261724184) |

- **stage-b-test-2-npu-a3**: 作业在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回418错误，无法获取文件，属于网络或代理环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723487

- **stage-b-test-1-npu-a3**: 作业在安装依赖时，通过gh-proxy代理下载sgl-kernel-npu zip包，服务器返回418错误（可能是代理拒绝或限流），导致下载失败，进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723522

- **stage-b-test-8-npu-a3**: 在安装依赖阶段，通过gh-proxy.test.osinfra.cn代理下载sgl-kernel-npu包时，代理服务器返回HTTP 418错误，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723532

- **stage-b-test-16-npu-a3**: 在安装依赖阶段，从gh-proxy.test.osinfra.cn下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时返回418错误，可能是代理或资源不可用，导致安装中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723565

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js版本弃用警告和上传artifact时未找到失败文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723571

- **multimodal-gen-test-2-npu-a3**: 日志中无测试失败或错误信息，仅包含Node.js 20弃用警告和未找到diffusion-failures文件的提示，作业流程正常完成。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723576

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业初始化和清理信息，未显示测试执行或错误输出，可能因环境准备失败或资源分配问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261724168

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在下载actions和设置环境后，于11:48:42开始执行plog备份和k8s脚本，随后进入清理阶段，未出现测试执行或失败信息，疑似runner环境异常导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261724184

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| set-image-config | 0.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261499096) |
| stage-b-test-4-npu-a3 (0) | 33.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723533) |
| stage-b-test-4-npu-a3 (1) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723572) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30992447131/job/92261723680) |


## [Run #30992000915](https://github.com/sgl-project/sglang/actions/runs/30992000915)
- **分支**: `pcg_conflict_merge`
- **总耗时**: 170.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30992000915

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 4.5min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153498) |
| stage-b-test-16-npu-a3 | 2.0min | 环境问题 | Git 拉取代码时代理返回 418 错误，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153506) |
| stage-b-test-1-npu-a3 | 4.5min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153518) |
| multimodal-gen-test-2-npu-a3 | 26.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153529) |
| stage-b-test-4-npu-a3 (0) | 2.9min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153583) |
| stage-b-test-2-npu-a3 | 4.7min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153584) |
| stage-b-test-4-npu-a3 (1) | 2.2min | 环境问题 | 下载sgl-kernel-npu依赖时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153592) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 4.5min | 其他 | 日志被截断，未显示测试执行结果，仅看到作业清理和Node.js弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153925) |

- **stage-b-test-8-npu-a3**: 作业在下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，代理服务器返回418错误，无法获取依赖包，导致构建流程中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153498

- **stage-b-test-16-npu-a3**: 作业在 actions/checkout 阶段通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 时，代理返回 HTTP 418，重试三次均失败，最终作业退出。属于代理或网络环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153506

- **stage-b-test-1-npu-a3**: 在安装依赖阶段，从gh-proxy.test.osinfra.cn下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时返回418错误，可能是代理或网络问题，导致下载失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153518

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node.js弃用警告及上传diffusion-failures目录时未找到文件，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153529

- **stage-b-test-4-npu-a3 (0)**: 在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载zip包，服务器返回418错误，下载失败，导致作业退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153583

- **stage-b-test-2-npu-a3**: 在安装sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，通过gh-proxy.test.osinfra.cn代理下载返回HTTP 418，可能是代理拒绝或资源不可用，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153584

- **stage-b-test-4-npu-a3 (1)**: CI在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release文件，但代理返回HTTP 418错误，导致下载失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153592

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志中间部分被省略，无法定位具体失败原因。仅能看到作业在约4分钟后进入清理阶段，且无metrics.json生成，可能测试未正常运行或提前退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153925

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153605) |
| stage-a-unit-test-npu | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153677) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30992000915/job/92260153829) |


## [Run #30991431202](https://github.com/sgl-project/sglang/actions/runs/30991431202)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 20.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30991431202

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 2.8min | 环境问题 | 下载依赖包时代理返回HTTP 418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418684) |
| multimodal-gen-test-1-npu-a3 | 19.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418743) |
| stage-b-test-16-npu-a3 | 19.3min | 环境问题 | NPU作业因Scheduler watchdog超时及容器执行失败而终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418745) |
| stage-b-test-1-npu-a3 | 2.3min | 环境问题 | 下载依赖包时代理返回HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418807) |
| stage-a-unit-test-npu | 19.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418808) |
| stage-b-test-2-npu-a3 | 2.3min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418817) |
| multimodal-gen-test-2-npu-a3 | 19.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418829) |
| stage-b-test-4-npu-a3 (1) | 13.9min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418848) |
| stage-b-test-4-npu-a3 (0) | 3.9min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418895) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 19.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258419456) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.4min | 其他 | 作业在启动阶段即失败，未进入实际测试，日志中无测试执行信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258419567) |

- **stage-b-test-8-npu-a3**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回418错误，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418684

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储资源缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418743

- **stage-b-test-16-npu-a3**: 日志显示在加载MoE模型权重时发生Scheduler watchdog timeout（300秒），随后自定义容器执行失败，导致作业中断。可能是NPU资源竞争或模型加载过慢引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418745

- **stage-b-test-1-npu-a3**: 在安装sgl-kernel-npu依赖时，通过gh-proxy.test.osinfra.cn代理下载custom-ops压缩包，服务器返回418错误，下载失败，导致作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418807

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418808

- **stage-b-test-2-npu-a3**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，但代理返回HTTP 418错误，下载失败，导致进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418817

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业依赖的某个文件（如模型权重或测试数据）在存储中缺失，可能是文件被删除、路径错误或上传未完成，需检查CI配置中的资源引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418829

- **stage-b-test-4-npu-a3 (1)**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员。日志显示NPU内存正常（约60GB），但容器在加载shards时崩溃，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418848

- **stage-b-test-4-npu-a3 (0)**: 作业在下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，通过gh-proxy.test.osinfra.cn代理访问GitHub，但代理返回HTTP 418错误，导致下载失败，进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258418895

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是资源被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258419456

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试脚本，可能因环境初始化失败或作业被提前终止，需查看更完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30991431202/job/92258419567


## [Run #30990991048](https://github.com/sgl-project/sglang/actions/runs/30990991048)
- **分支**: `main`
- **总耗时**: 58.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30990991048

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 2.4min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906898) |
| stage-b-test-16-npu-a3 | 1.9min | 环境问题 | GitHub Actions 拉取代码时代理返回 418 错误，重试后成功。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906916) |
| stage-b-test-4-npu-a3 (0) | 2.7min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906931) |
| stage-b-test-8-npu-a3 | 2.5min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906975) |
| multimodal-gen-test-1-npu-a3 | 58.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906993) |
| stage-a-unit-test-npu | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907006) |
| multimodal-gen-test-2-npu-a3 | 58.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907022) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907278) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.6min | 环境问题 | 作业在启动阶段即失败，未执行实际测试，缺少关键错误日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907459) |

- **stage-b-test-2-npu-a3**: 在安装依赖阶段，通过gh-proxy代理下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，代理返回418错误，下载失败，进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906898

- **stage-b-test-16-npu-a3**: checkout 阶段通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 时首次返回 418，等待 16 秒重试后成功，属于临时性网络/代理故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906916

- **stage-b-test-4-npu-a3 (0)**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，代理返回HTTP 418错误，下载失败，导致整个作业退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906931

- **stage-b-test-8-npu-a3**: 作业在安装sgl-kernel-npu时，通过gh-proxy代理下载GitHub release包，服务器返回418错误，导致下载失败，进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906975

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问Azure Blob存储中的文件时，返回BlobNotFound错误，说明所需文件缺失或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906993

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907006

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置变更，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907022

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明作业依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查资源是否存在或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907278

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试用例，且无metrics.json生成，可能因环境初始化失败或资源分配问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907459

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a3 | 51.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256906918) |
| stage-b-test-4-npu-a3 (1) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990991048/job/92256907023) |


## [Run #30990926321](https://github.com/sgl-project/sglang/actions/runs/30990926321)
- **分支**: `fix/rope-config-and-vl-weight-loading`
- **总耗时**: 169.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30990926321

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 2.5min | 环境问题 | 下载依赖包时代理返回HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720234) |
| multimodal-gen-test-2-npu-a3 | 19.2min | 其他 | 作业日志不完整，未显示实际测试命令和失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720270) |
| stage-b-test-4-npu-a3 (1) | 1.7min | 环境问题 | Git 代理拉取仓库时返回 418 错误，重试后成功，但作业可能因此中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720294) |
| stage-b-test-4-npu-a3 (0) | 2.3min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720347) |
| stage-b-test-16-npu-a3 | 3.0min | 环境问题 | 下载依赖包时代理返回HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720458) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.1min | 环境问题 | Git 代理访问 GitHub 仓库时返回 418 错误，导致代码拉取失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256721007) |

- **stage-b-test-8-npu-a3**: 在下载ops-transformer-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回418错误，无法获取文件，属于网络或代理环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720234

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行部分，仅显示runner启动、依赖下载和上传工件（无文件）。可能因日志截断或作业在测试前已失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720270

- **stage-b-test-4-npu-a3 (1)**: 日志显示 git fetch 通过代理 gh-proxy.test.osinfra.cn 访问 GitHub 时首次返回 418 错误，重试后成功。该错误属于临时性网络/代理问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720294

- **stage-b-test-4-npu-a3 (0)**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，但代理返回HTTP 418错误，下载失败，导致整个作业退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720347

- **stage-b-test-16-npu-a3**: 在安装sgl-kernel-npu依赖时，从gh-proxy.test.osinfra.cn下载custom-ops zip包时服务器返回418错误，下载失败，导致整个CI作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720458

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 作业在 checkout 阶段通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 仓库时，多次收到 HTTP 418 错误，重试后虽成功，但已造成延迟，可能影响后续任务执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256721007

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a3 | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720235) |
| multimodal-gen-test-1-npu-a3 | 27.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720278) |
| stage-b-test-1-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720293) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256720384) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990926321/job/92256721114) |


## [Run #30990412670](https://github.com/sgl-project/sglang/actions/runs/30990412670)
- **分支**: `xinyuan/parser-auto-resolution-order`
- **总耗时**: 108.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30990412670

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (1) | 1.4min | 环境问题 | 作业在checkout后立即结束，无实际测试日志，疑似基础设施问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118669) |
| stage-b-test-4-npu-a3 (0) | 2.2min | 环境问题 | 下载依赖包时代理返回418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118675) |
| stage-b-test-8-npu-a3 | 2.1min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118676) |
| stage-b-test-2-npu-a3 | 1.9min | 环境问题 | 下载sgl-kernel-npu依赖时HTTP 418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118678) |
| stage-b-test-1-npu-a3 | 2.1min | 环境问题 | 下载依赖包时代理返回418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118681) |
| stage-b-test-16-npu-a3 | 3.2min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118745) |
| multimodal-gen-test-2-npu-a3 | 26.3min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118779) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.5min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到作业启动和清理过程。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255119165) |

- **stage-b-test-4-npu-a3 (1)**: 日志显示actions/checkout成功拉取PR #33485代码后，作业直接进入清理阶段，未执行任何测试步骤。可能是runner环境异常、作业被外部取消或调度问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118669

- **stage-b-test-4-npu-a3 (0)**: 在安装sgl-kernel-npu依赖时，从gh-proxy.test.osinfra.cn下载custom-ops包时服务器返回HTTP 418错误，下载失败，导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118675

- **stage-b-test-8-npu-a3**: 在安装依赖阶段，通过gh-proxy.test.osinfra.cn代理下载sgl-kernel-npu压缩包时，代理服务器返回HTTP 418错误，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118676

- **stage-b-test-2-npu-a3**: pip安装依赖后，尝试从gh-proxy.test.osinfra.cn下载sgl-kernel-npu包时返回418错误，可能是代理或网络问题，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118678

- **stage-b-test-1-npu-a3**: 在下载ops-transformer-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回HTTP 418错误，文件无法获取，导致CI流程中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118681

- **stage-b-test-16-npu-a3**: 在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，服务器返回HTTP 418错误，下载失败，导致作业退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118745

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118779

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志中间部分被省略，无法看到测试执行细节。作业在启动后很快进入清理阶段，可能因环境初始化失败或测试未实际运行而提前结束，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255119165

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118687) |
| stage-a-unit-test-npu | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255118696) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30990412670/job/92255119161) |


## [Run #30990132805](https://github.com/sgl-project/sglang/actions/runs/30990132805)
- **分支**: `glm-image-usage-report`
- **总耗时**: 100.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30990132805

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 100.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990132805/job/92262665579) |
| multimodal-gen-test-2-npu-a3 | 100.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30990132805/job/92262665625) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或数据在 Azure Blob 存储中缺失，可能是资源被清理或路径错误，需检查相关配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990132805/job/92262665579

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误，或资源已被删除。属于环境配置或依赖资源问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30990132805/job/92262665625


## [Run #30989747392](https://github.com/sgl-project/sglang/actions/runs/30989747392)
- **分支**: `main`
- **总耗时**: 6.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30989747392

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 0.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001282) |
| stage-b-test-4-npu-a3 (1) | 0.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001316) |
| stage-b-test-16-npu-a3 | 0.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001343) |
| stage-b-test-2-npu-a3 | 0.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001345) |
| stage-a-unit-test-npu | 0.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001351) |
| stage-b-test-4-npu-a3 (0) | 0.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001390) |
| stage-b-test-1-npu-a3 | 0.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001414) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 0.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001614) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 5.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001625) |

- **stage-b-test-8-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001282

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或存储配置变更，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001316

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001343

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于环境或资源配置问题，需检查相关存储路径或文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001345

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001351

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001390

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001414

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001614

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或数据集）已被删除或路径错误，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30989747392/job/92253001625


## [Run #30988801740](https://github.com/sgl-project/sglang/actions/runs/30988801740)
- **分支**: `pr-wan-vae-norm-silu-quality-high`
- **总耗时**: 134.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30988801740

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 30.6min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249847963) |
| stage-b-test-8-npu-a3 | 2.2min | 环境问题 | Git 拉取代码时代理返回 418 错误，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249847974) |
| stage-b-test-4-npu-a3 (0) | 2.5min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848004) |
| multimodal-gen-test-2-npu-a3 | 31.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848023) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 1.6min | 环境问题 | Pod启动失败，容器因OOM被SIGKILL终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848355) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未发现测试执行或失败记录，可能因前置条件未满足或作业被跳过，需检查完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249847963

- **stage-b-test-8-npu-a3**: 作业在 actions/checkout 阶段执行 git fetch 时，通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 仓库，代理返回 HTTP 418 错误，重试三次均失败，最终作业退出。属于代理或网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249847974

- **stage-b-test-4-npu-a3 (0)**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，服务器返回HTTP 418错误，下载失败，导致整个流程退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848004

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848023

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 作业在启动阶段即失败，容器进程被强制终止（exit code 137），提示内存溢出或资源限制，属于基础设施环境问题，非测试代码或模型精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848355

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 62.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249847960) |
| stage-b-test-2-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249847982) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249847994) |
| stage-b-test-4-npu-a3 (1) | 16.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848044) |
| stage-b-test-1-npu-a3 | 26.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848056) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30988801740/job/92249848335) |


## [Run #30988439922](https://github.com/sgl-project/sglang/actions/runs/30988439922)
- **分支**: `xinyuan/parser-auto-resolution-order`
- **总耗时**: 29.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30988439922

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (0) | 3.1min | 环境问题 | 下载sgl-kernel-npu依赖时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699184) |
| stage-b-test-16-npu-a3 | 18.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699205) |
| stage-b-test-1-npu-a3 | 18.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699214) |
| multimodal-gen-test-1-npu-a3 | 28.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699219) |
| stage-b-test-4-npu-a3 (1) | 9.6min | 环境问题 | 自定义容器执行失败，NPU后端算子回退CPU导致性能下降 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699251) |
| stage-a-unit-test-npu | 28.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699299) |
| multimodal-gen-test-2-npu-a3 | 28.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699300) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 28.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699707) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 20.6min | 其他 | 日志被截断，未显示测试执行结果，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699798) |

- **stage-b-test-4-npu-a3 (0)**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载zip包，服务器返回HTTP 418错误，下载失败，导致整个作业退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699184

- **stage-b-test-16-npu-a3**: 作业在启动NPU测试时，自定义容器实现执行失败，TokenizerManager初始化后容器崩溃，可能是NPU资源或环境配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699205

- **stage-b-test-1-npu-a3**: 测试运行到第5个用例时，自定义容器实现执行失败，导致作业中断。日志显示NPU环境或容器配置存在问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699214

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699219

- **stage-b-test-4-npu-a3 (1)**: 作业在NPU测试过程中，多个算子（如aten::_assert_async）不支持NPU后端回退到CPU执行，导致性能下降，最终自定义容器实现执行失败，属于环境/硬件兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699251

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699299

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699300

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699707

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业初始化和清理阶段，未展示测试运行输出或错误信息，可能因日志截断或作业在早期阶段异常终止，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699798

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 8.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699227) |
| stage-b-test-2-npu-a3 | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30988439922/job/92248699250) |


## [Run #30987819364](https://github.com/sgl-project/sglang/actions/runs/30987819364)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 52.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30987819364

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 36.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840538) |
| stage-b-test-4-npu-a3 (1) | 3.0min | 环境问题 | 下载依赖包时代理返回418错误，导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840543) |
| multimodal-gen-test-1-npu-a3 | 51.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840550) |
| multimodal-gen-test-2-npu-a3 | 51.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840554) |
| stage-b-test-2-npu-a3 | 2.7min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误，导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840559) |
| stage-b-test-1-npu-a3 | 3.1min | 环境问题 | 下载sgl-kernel-npu依赖时代理返回418错误，导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840571) |
| stage-b-test-4-npu-a3 (0) | 2.6min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840658) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 51.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840971) |

- **stage-b-test-16-npu-a3**: 日志显示模型分片加载至87%时，自定义容器实现执行失败，提示联系自托管runner管理员，属基础设施环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840538

- **stage-b-test-4-npu-a3 (1)**: 在下载ops-transformer-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回HTTP 418错误，文件无法下载，导致CI流程中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840543

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840550

- **multimodal-gen-test-2-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840554

- **stage-b-test-2-npu-a3**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载release包，服务器返回418错误，下载失败导致进程退出。属于网络/代理环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840559

- **stage-b-test-1-npu-a3**: 作业在下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，通过gh-proxy.test.osinfra.cn代理访问GitHub，服务器返回HTTP 418错误，导致下载失败，进程退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840571

- **stage-b-test-4-npu-a3 (0)**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，但代理返回HTTP 418错误，下载失败，导致整个作业退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840658

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到。可能是 CI 配置引用了不存在的文件，或存储被清理/路径错误。建议检查作业依赖的 blob 路径是否正确，或确认存储账户状态。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840971

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840510) |
| stage-b-test-8-npu-a3 | 8.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840641) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30987819364/job/92246840913) |


## [Run #30987483567](https://github.com/sgl-project/sglang/actions/runs/30987483567)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 5.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30987483567

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 2.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245975956) |
| multimodal-gen-test-2-npu-a3 | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245975994) |
| stage-b-test-4-npu-a3 (0) | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976019) |
| stage-b-test-4-npu-a3 (1) | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976025) |
| stage-b-test-8-npu-a3 | 2.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976038) |
| multimodal-gen-test-1-npu-a3 | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976055) |
| stage-b-test-1-npu-a3 | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976073) |
| stage-a-unit-test-npu | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976156) |
| stage-b-test-2-npu-a3 | 2.3min | 其他 | 作业在准备阶段即被终止，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976239) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976594) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976639) |

- **stage-b-test-16-npu-a3**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被删除、路径错误或上传未完成。这属于外部依赖环境问题，需检查CI配置中的blob路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245975956

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245975994

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976019

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976025

- **stage-b-test-8-npu-a3**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976038

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976055

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976073

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976156

- **stage-b-test-2-npu-a3**: 日志显示作业在checkout完成后，执行k8s/index.js时被清理（Cleaning up orphan processes），可能是runner被外部中断或资源回收，无测试相关错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976239

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976594

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987483567/job/92245976639


## [Run #30987102310](https://github.com/sgl-project/sglang/actions/runs/30987102310)
- **分支**: `pr-wan-vae-norm-silu-quality-high`
- **总耗时**: 24.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30987102310

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 1.6min | 环境问题 | GitHub Actions 拉取代码时代理返回 418 错误，重试后成功，但首次失败导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383804) |
| stage-b-test-16-npu-a3 | 14.3min | 环境问题 | 下载依赖包时代理返回418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383823) |
| multimodal-gen-test-1-npu-a3 | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383842) |
| stage-b-test-4-npu-a3 (0) | 11.8min | 环境问题 | 下载sgl-kernel-npu依赖时代理返回418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383851) |
| stage-b-test-2-npu-a3 | 10.9min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383864) |
| multimodal-gen-test-2-npu-a3 | 24.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383867) |
| stage-b-test-1-npu-a3 | 7.5min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383885) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 10.9min | 环境问题 | GitHub Actions 拉取代码时网络代理返回 418 错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244384095) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244384130) |
| stage-b-test-4-npu-a3 (1) | 11.4min | 环境问题 | 下载依赖包时代理返回418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244384139) |

- **stage-b-test-8-npu-a3**: checkout 阶段通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 时返回 HTTP 418，git fetch 失败退出码 128，重试后成功。属于代理服务临时故障，非代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383804

- **stage-b-test-16-npu-a3**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回HTTP 418错误，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383823

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383842

- **stage-b-test-4-npu-a3 (0)**: CI在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，但代理返回HTTP 418错误，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383851

- **stage-b-test-2-npu-a3**: 作业在安装sgl-kernel-npu时，通过gh-proxy代理下载GitHub release包返回418错误，可能是代理被拒绝或资源不可用，导致安装失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383864

- **multimodal-gen-test-2-npu-a3**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383867

- **stage-b-test-1-npu-a3**: 作业在安装依赖时，通过gh-proxy代理下载sgl-kernel-npu压缩包，服务器返回418错误（拒绝访问），可能是代理或网络问题，导致下载失败，进程退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383885

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 作业在 checkout 阶段通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 仓库时，代理返回 HTTP 418 错误，重试三次均失败，最终退出码 1。属于网络/代理环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244384095

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244384130

- **stage-b-test-4-npu-a3 (1)**: 在下载ops-transformer zip包时，gh-proxy.test.osinfra.cn代理返回HTTP 418错误，导致下载失败，作业退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244384139

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30987102310/job/92244383848) |


## [Run #30987099095](https://github.com/sgl-project/sglang/actions/runs/30987099095)
- **分支**: `new_layernorm`
- **总耗时**: 156.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30987099095

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 30.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30987099095/job/92244397828) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前结束，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30987099095/job/92244397828

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30987099095/job/92244397835) |


## [Run #30986619617](https://github.com/sgl-project/sglang/actions/runs/30986619617)
- **分支**: `fuse-gate-gemv-into-append`
- **总耗时**: 132.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30986619617

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 12.3min | 环境问题 | 下载依赖包时代理返回418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845588) |
| stage-b-test-1-npu-a3 | 13.4min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845589) |
| stage-b-test-8-npu-a3 | 7.4min | 环境问题 | 下载依赖包时代理返回418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845601) |
| multimodal-gen-test-2-npu-a3 | 26.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845766) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 15.3min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242846247) |

- **stage-b-test-2-npu-a3**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回HTTP 418错误，无法获取文件，导致流程中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845588

- **stage-b-test-1-npu-a3**: 作业在下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回HTTP 418错误，无法获取文件，最终进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845589

- **stage-b-test-8-npu-a3**: 在安装sgl-kernel-npu依赖时，通过gh-proxy.test.osinfra.cn下载custom-ops包时服务器返回HTTP 418错误，可能是代理限制或临时故障，导致下载失败并退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845601

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions环境准备、Node版本警告及上传artifact时未找到文件的提示，无法判断测试失败的具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845766

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志在测试运行前中断，未包含实际测试输出或错误信息，无法判断失败原因。可能为基础设施问题或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242846247

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 52.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845594) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845620) |
| stage-b-test-4-npu-a3 (1) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845686) |
| stage-b-test-4-npu-a3 (0) | 33.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845728) |
| multimodal-gen-test-1-npu-a3 | 27.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242845733) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986619617/job/92242846261) |


## [Run #30986609155](https://github.com/sgl-project/sglang/actions/runs/30986609155)
- **分支**: `feat/graceful-shutdown`
- **总耗时**: 106.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30986609155

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 5.9min | 环境问题 | 下载依赖包时代理返回HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855700) |
| stage-b-test-4-npu-a3 (0) | 14.5min | 环境问题 | 下载sgl-kernel-npu依赖时HTTP 418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855709) |
| stage-b-test-2-npu-a3 | 9.4min | 环境问题 | Git 代理访问 GitHub 仓库时返回 418 错误，重试后成功，但作业最终失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855720) |
| multimodal-gen-test-2-npu-a3 | 23.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855749) |
| stage-b-test-4-npu-a3 (1) | 14.9min | 环境问题 | 下载依赖包时代理返回418错误，导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855793) |
| stage-b-test-8-npu-a3 | 2.3min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855794) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 3.3min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242856127) |

- **stage-b-test-16-npu-a3**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回418错误，无法获取文件，属于外部网络或代理环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855700

- **stage-b-test-4-npu-a3 (0)**: 作业在安装sgl-kernel-npu时，通过gh-proxy代理下载zip包，服务器返回418错误，下载失败，导致进程退出码1，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855709

- **stage-b-test-2-npu-a3**: 在 checkout 阶段，git fetch 通过代理 gh-proxy.test.osinfra.cn 访问 GitHub 时首次返回 418 错误，重试后成功获取代码。但作业后续仍失败，可能是代理不稳定或网络环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855720

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js版本弃用警告和上传artifact时无文件提示，无法判断具体失败原因，可能为作业被提前终止或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855749

- **stage-b-test-4-npu-a3 (1)**: 在下载ops-transformer zip包时，gh-proxy.test.osinfra.cn代理返回HTTP 418错误，下载失败，导致CI流程中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855793

- **stage-b-test-8-npu-a3**: 日志显示在下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，代理服务器gh-proxy.test.osinfra.cn返回418错误，可能是代理拒绝或资源不可用，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855794

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业初始化和清理阶段信息，未展示测试运行过程及错误输出，需查看完整日志以定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242856127

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855683) |
| stage-b-test-1-npu-a3 | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855696) |
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242855829) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986609155/job/92242856162) |


## [Run #30986537196](https://github.com/sgl-project/sglang/actions/runs/30986537196)
- **分支**: `fix-mistral-native-detect-unsharded`
- **总耗时**: 65.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30986537196

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 3.1min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571139) |
| stage-b-test-2-npu-a3 | 1.5min | 其他 | 日志不完整，仅显示checkout成功，无实际测试执行信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571150) |
| stage-b-test-4-npu-a3 (0) | 1.9min | 环境问题 | Git 代理访问 GitHub 返回 418 错误，导致代码拉取失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571155) |
| multimodal-gen-test-1-npu-a3 | 4.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571158) |
| stage-b-test-4-npu-a3 (1) | 2.6min | 环境问题 | 下载依赖包时代理返回HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571187) |
| stage-b-test-1-npu-a3 | 2.2min | 环境问题 | 下载依赖包时代理返回HTTP 418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571191) |
| stage-b-test-16-npu-a3 | 12.0min | 环境问题 | 下载 sgl-kernel-npu 依赖时代理返回 418 错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571196) |
| multimodal-gen-test-2-npu-a3 | 64.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571224) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 1.4min | 环境问题 | 作业在初始化阶段即失败，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571566) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 64.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571571) |

- **stage-b-test-8-npu-a3**: 作业在安装依赖时，通过gh-proxy代理下载sgl-kernel-npu压缩包，服务器返回418错误，下载失败，导致流程退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571139

- **stage-b-test-2-npu-a3**: 日志在checkout完成后即结束，未包含任何测试运行、编译或错误信息，无法判断具体失败原因，可能为日志截断或作业被外部中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571150

- **stage-b-test-4-npu-a3 (0)**: checkout 阶段通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 时返回 418，重试后成功，但首次失败可能影响作业稳定性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571155

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前结束，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571158

- **stage-b-test-4-npu-a3 (1)**: 在安装sgl-kernel-npu依赖时，从gh-proxy.test.osinfra.cn下载custom-ops zip包时服务器返回418错误，下载失败，导致流程中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571187

- **stage-b-test-1-npu-a3**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回418错误，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571191

- **stage-b-test-16-npu-a3**: 作业在安装 sgl-kernel-npu 时，通过 gh-proxy.test.osinfra.cn 代理下载 GitHub 资源，但代理返回 HTTP 418，下载失败，最终进程退出码 1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571196

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件或数据在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571224

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在checkout后启动k8s index.js时中断，仅有Node.js 20弃用警告，无测试执行或错误信息，疑似基础设施或调度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571566

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571571

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30986537196/job/92242571130) |


---
*Auto-generated by npu_pr_monitor.py*