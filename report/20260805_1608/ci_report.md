# NPU CI 执行监控
**生成时间**: 2026-08-05 08:08 UTC
**分析 Run 数**: 15

---

## [Run #30984624927](https://github.com/sgl-project/sglang/actions/runs/30984624927)
- **分支**: `t4-ulysses-a2a-pack`
- **总耗时**: 21.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30984624927

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 13.2min | 环境问题 | 自定义容器执行失败，NPU后端不支持CUDA相关操作导致服务异常退出 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538754) |
| stage-b-test-1-npu-a3 | 3.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538769) |
| stage-b-test-8-npu-a3 | 2.2min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538788) |
| stage-b-test-16-npu-a3 | 1.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538789) |
| stage-b-test-4-npu-a3 (1) | 2.2min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538819) |
| stage-b-test-4-npu-a3 (0) | 3.5min | 环境问题 | 自定义容器执行失败，测试启动后立即中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538835) |
| multimodal-gen-test-2-npu-a3 | 20.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538843) |
| multimodal-gen-test-1-npu-a3 | 20.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538858) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 1.9min | 环境问题 | GitHub Actions 拉取仓库时代理返回 418 错误，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236539217) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 20.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236539269) |

- **stage-b-test-2-npu-a3**: 日志显示SymmetricMemory不支持cuda设备类型，且aten::_assert_async算子回退到CPU执行，最终自定义容器实现执行失败，属于NPU环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538754

- **stage-b-test-1-npu-a3**: 作业在启动第一个NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538769

- **stage-b-test-8-npu-a3**: 作业在安装Rust工具链时，自定义容器实现执行失败（Executing the custom container implementation failed），属于runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538788

- **stage-b-test-16-npu-a3**: 在安装triton-ascend依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538789

- **stage-b-test-4-npu-a3 (1)**: 在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载zip包，服务器返回HTTP 418错误，下载失败，导致作业退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538819

- **stage-b-test-4-npu-a3 (0)**: 作业在运行第一个测试test_npu_hicache_mla.py时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538835

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误，或资源已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538843

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业依赖的某个blob文件缺失或路径错误，可能是上传失败、文件被删除或配置的URL有误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538858

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 作业在 fetch 仓库阶段，通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 时返回 HTTP 418，重试三次均失败，最终退出码 1。属于代理或网络环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236539217

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或数据文件未上传或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236539269

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30984624927/job/92236538926) |


## [Run #30984310044](https://github.com/sgl-project/sglang/actions/runs/30984310044)
- **分支**: `t1-sm100-cudnn-sdpa`
- **总耗时**: 22.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30984310044

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 21.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984310044/job/92235632633) |
| multimodal-gen-test-2-npu-a3 | 21.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30984310044/job/92235632634) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或数据在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984310044/job/92235632633

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未上传或路径错误，属于环境配置或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30984310044/job/92235632634


## [Run #30983841597](https://github.com/sgl-project/sglang/actions/runs/30983841597)
- **分支**: `kan/rust-server-native-mm`
- **总耗时**: 34.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30983841597

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 2.6min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234099969) |
| stage-b-test-2-npu-a3 | 2.3min | 环境问题 | 下载依赖包时代理返回HTTP 418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234099970) |
| multimodal-gen-test-2-npu-a3 | 33.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100037) |
| stage-b-test-4-npu-a3 (0) | 2.2min | 环境问题 | 下载sgl-kernel-npu依赖时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100040) |
| stage-b-test-16-npu-a3 | 2.7min | 环境问题 | 下载依赖包时代理返回418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100053) |
| multimodal-gen-test-1-npu-a3 | 33.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100060) |
| stage-b-test-4-npu-a3 (1) | 2.7min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100065) |
| stage-b-test-1-npu-a3 | 18.7min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100135) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 33.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100593) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 7.0min | 其他 | 日志被截断，未显示测试执行结果，无法确定失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100605) |

- **stage-b-test-8-npu-a3**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，但代理返回HTTP 418错误，下载失败，导致整个作业以退出码1终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234099969

- **stage-b-test-2-npu-a3**: 在下载ops-transformer-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回418错误，导致下载失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234099970

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100037

- **stage-b-test-4-npu-a3 (0)**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载zip包，服务器返回418错误，下载失败，导致整个作业退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100040

- **stage-b-test-16-npu-a3**: 作业在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回HTTP 418错误，导致下载失败，作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100053

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传，或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100060

- **stage-b-test-4-npu-a3 (1)**: 日志显示pip安装依赖后，通过gh-proxy.test.osinfra.cn代理下载sgl-kernel-npu包时，服务器返回HTTP 418错误，下载失败，导致作业退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100065

- **stage-b-test-1-npu-a3**: 日志显示模型权重加载成功，但在后续执行时出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100135

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100593

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业初始化和清理阶段，未展示测试运行输出或错误信息，可能因日志截断导致信息缺失，需查看完整日志以定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100605

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 9.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30983841597/job/92234100008) |


## [Run #30983329680](https://github.com/sgl-project/sglang/actions/runs/30983329680)
- **分支**: `mmangkad/fix-serving-benchmark-cache-flush-race`
- **总耗时**: 6.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30983329680

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907016) |
| stage-b-test-1-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907040) |
| stage-b-test-16-npu-a3 | 6.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907047) |
| stage-b-test-8-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907075) |
| stage-b-test-4-npu-a3 (1) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907100) |
| stage-b-test-4-npu-a3 (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907138) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907699) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907718) |

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907016

- **stage-b-test-1-npu-a3**: 作业 stage-b-test-1-npu-a3 在尝试下载或访问 Azure Blob 中的某个 blob 时，返回 BlobNotFound 错误（HTTP 404）。这通常是因为日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907040

- **stage-b-test-16-npu-a3**: 作业在尝试访问Azure Blob存储时，因指定的blob不存在（BlobNotFound）而失败。这可能是由于文件被删除、路径错误或上传未完成，属于环境或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907047

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907075

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907100

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907138

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907699

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907718

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30983329680/job/92233907026) |


## [Run #30979644109](https://github.com/sgl-project/sglang/actions/runs/30979644109)
- **分支**: `t1-sm100-cudnn-sdpa`
- **总耗时**: 79.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30979644109

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30979644109/job/92221181866) |
| multimodal-gen-test-1-npu-a3 | 79.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30979644109/job/92221181876) |

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30979644109/job/92221181866

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或数据在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30979644109/job/92221181876


## [Run #30979181335](https://github.com/sgl-project/sglang/actions/runs/30979181335)
- **分支**: `main`
- **总耗时**: 33.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30979181335

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 32.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30979181335/job/92219812065) |
| multimodal-gen-test-1-npu-a3 | 32.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30979181335/job/92219812085) |

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30979181335/job/92219812065

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中缺失，可能是资源未上传或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30979181335/job/92219812085


## [Run #30978922956](https://github.com/sgl-project/sglang/actions/runs/30978922956)
- **分支**: `mtp-draft-sidecar-pools`
- **总耗时**: 67.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30978922956

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 (1) | 2.3min | 环境问题 | 下载依赖包时代理返回HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041879) |
| multimodal-gen-test-2-npu-a3 | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041898) |
| stage-b-test-4-npu-a3 (0) | 1.2min | 环境问题 | 作业在checkout后立即结束，未执行实际测试，疑似runner环境异常或作业被提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041903) |
| stage-b-test-16-npu-a3 | 1.1min | 环境问题 | GitHub Actions 运行器使用已弃用的 Node.js 20，被强制升级到 Node.js 24，导致兼容性警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041904) |
| multimodal-gen-test-1-npu-a3 | 66.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041916) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 66.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219042378) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 0.7min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219042392) |

- **stage-b-test-4-npu-a3 (1)**: 作业在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回418错误，无法获取文件，属于网络或代理环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041879

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041898

- **stage-b-test-4-npu-a3 (0)**: 日志显示checkout成功（HEAD at 70d9e8e），随后进入清理进程阶段，无任何测试输出。可能是runner节点问题、资源分配失败或作业被外部取消，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041903

- **stage-b-test-16-npu-a3**: 作业在 checkout 阶段因 Node.js 20 弃用而触发警告，但未显示实际失败错误，可能因环境配置问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041904

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志文件或依赖数据未正确上传，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041916

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及权限。
  链接: https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219042378

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: Runner 在执行自定义容器实现时失败，提示 jobPod 未设置，说明 prepareJob 未成功完成，可能是 Kubernetes Pod 调度或容器启动异常，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219042392

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041876) |
| stage-a-unit-test-npu | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041889) |
| stage-b-test-1-npu-a3 | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041896) |
| stage-b-test-8-npu-a3 | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30978922956/job/92219041923) |


## [Run #30977609509](https://github.com/sgl-project/sglang/actions/runs/30977609509)
- **分支**: `ds_v4_xpu_fused_q_norm_rope`
- **总耗时**: 137.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30977609509

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 1.9min | 环境问题 | GitHub Actions 拉取代码时代理返回 418 错误，重试后成功。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925229) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.7min | 其他 | 日志被截断，未显示实际测试结果，无法判断失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925520) |

- **stage-b-test-16-npu-a3**: 作业在 checkout 阶段通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 时首次返回 418 错误，导致 git fetch 失败，但重试后成功，属于临时性网络/代理环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925229

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业启动和清理信息，未展示测试执行过程及失败点，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925520

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925209) |
| stage-b-test-8-npu-a3 | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925210) |
| stage-b-test-4-npu-a3 (0) | 33.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925212) |
| stage-b-test-2-npu-a3 | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925222) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925232) |
| stage-b-test-4-npu-a3 (1) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925242) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30977609509/job/92214925499) |


## [Run #30976930412](https://github.com/sgl-project/sglang/actions/runs/30976930412)
- **分支**: `ling3-flash-dspark`
- **总耗时**: 159.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30976930412

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 2.3min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873010) |
| multimodal-gen-test-2-npu-a3 | 17.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873063) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.5min | 环境问题 | 作业在准备阶段即失败，未进入实际测试，可能因运行环境或资源问题导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873296) |

- **stage-b-test-16-npu-a3**: 作业在下载sgl-kernel-npu-2026.7.27-torch2.10.0-py311-cann9.0.0-a3-aarch64.zip时，代理gh-proxy.test.osinfra.cn返回418错误，导致下载失败，进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873010

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873063

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在下载actions后立即进入清理阶段，未执行任何测试步骤，且无错误信息，可能因runner环境异常或资源分配失败导致提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873296

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873018) |
| stage-b-test-1-npu-a3 | 26.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873034) |
| stage-b-test-8-npu-a3 | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873039) |
| stage-b-test-4-npu-a3 (0) | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873042) |
| stage-b-test-4-npu-a3 (1) | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873057) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873064) |
| stage-b-test-2-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873068) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30976930412/job/92212873335) |


## [Run #30976370904](https://github.com/sgl-project/sglang/actions/runs/30976370904)
- **分支**: `ling3-flash-dspark`
- **总耗时**: 11.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30976370904

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 10.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153062) |
| multimodal-gen-test-1-npu-a3 | 10.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153085) |
| stage-b-test-16-npu-a3 | 10.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153088) |
| stage-b-test-1-npu-a3 | 10.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153089) |
| stage-b-test-4-npu-a3 (0) | 10.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153100) |
| stage-a-unit-test-npu | 7.9min | 环境问题 | NPU测试执行时自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153103) |
| stage-b-test-8-npu-a3 | 10.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153114) |
| multimodal-gen-test-2-npu-a3 | 10.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153143) |
| stage-b-test-4-npu-a3 (1) | 10.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153174) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153418) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 10.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153440) |

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153062

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153085

- **stage-b-test-16-npu-a3**: 作业在下载或访问Azure Blob存储中的某个文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153088

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153089

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153100

- **stage-a-unit-test-npu**: 作业在运行NPU单元测试时，自定义容器执行失败（Executing the custom container implementation failed），可能是NPU环境或容器配置问题，测试未实际运行即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153103

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153114

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153143

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，需检查 blob 名称或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153174

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到。可能是 CI 配置引用了不存在的文件，或存储被清理/路径错误，需检查相关 blob 路径或上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153418

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30976370904/job/92211153440


## [Run #30975937949](https://github.com/sgl-project/sglang/actions/runs/30975937949)
- **分支**: `ling3-flash-dspark`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30975937949

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317364) |
| stage-b-test-16-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317367) |
| stage-b-test-2-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317380) |
| stage-b-test-4-npu-a3 (1) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317392) |
| multimodal-gen-test-1-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317401) |
| stage-a-unit-test-npu | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317403) |
| multimodal-gen-test-2-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317412) |
| stage-b-test-1-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317417) |
| stage-b-test-4-npu-a3 (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317506) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317716) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317740) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317364

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317367

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317380

- **stage-b-test-4-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317392

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或数据文件未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317401

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317403

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317412

- **stage-b-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317417

- **stage-b-test-4-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317506

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317716

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传、被删除或配置有误，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975937949/job/92210317740


## [Run #30975241276](https://github.com/sgl-project/sglang/actions/runs/30975241276)
- **分支**: `fix/qwen35-mori-correct-routing-aiter-buffer`
- **总耗时**: 194.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30975241276

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 1.7min | 环境问题 | GitHub Actions 拉取代码时代理返回 418 错误，重试后成功，但作业已失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831565) |
| stage-b-test-16-npu-a3 | 2.5min | 环境问题 | 下载sgl-kernel-npu依赖时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831578) |
| multimodal-gen-test-2-npu-a3 | 26.7min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831580) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 5.5min | 其他 | 作业日志不完整，未显示实际失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831838) |

- **stage-b-test-8-npu-a3**: checkout 阶段通过 gh-proxy.test.osinfra.cn 代理访问 GitHub 时返回 418，重试两次后成功，但该错误导致作业标记为失败，属于代理或网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831565

- **stage-b-test-16-npu-a3**: 在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载zip包，服务器返回HTTP 418错误，下载失败，导致作业退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831578

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures目录，但未找到任何文件，未运行实际测试或出现明确错误，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831580

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志截断，缺少测试执行和失败断言部分，无法判断具体失败原因。可能为测试未运行或日志收集不完整，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831838

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 (0) | 34.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831574) |
| stage-b-test-1-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831587) |
| stage-b-test-2-npu-a3 | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831590) |
| stage-b-test-4-npu-a3 (1) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831599) |
| stage-a-unit-test-npu | 9.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831605) |
| multimodal-gen-test-1-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831662) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30975241276/job/92207831845) |


## [Run #30975146994](https://github.com/sgl-project/sglang/actions/runs/30975146994)
- **分支**: `refactor-rocm-deepseek-attention`
- **总耗时**: 21.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30975146994

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 2.4min | 环境问题 | 下载依赖包时代理返回HTTP 418错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742297) |
| stage-b-test-8-npu-a3 | 2.1min | 环境问题 | 下载依赖包时HTTP 418错误导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742299) |
| stage-b-test-4-npu-a3 (1) | 2.6min | 环境问题 | 下载依赖包时代理返回HTTP 418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742313) |
| stage-b-test-1-npu-a3 | 2.4min | 环境问题 | 下载sgl-kernel-npu依赖包时HTTP 418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742320) |
| stage-b-test-16-npu-a3 | 21.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742353) |
| multimodal-gen-test-1-npu-a3 | 21.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742358) |
| multimodal-gen-test-2-npu-a3 | 21.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742425) |
| stage-b-test-4-npu-a3 (0) | 2.5min | 环境问题 | 下载sgl-kernel-npu依赖包时代理返回HTTP 418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742459) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 21.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742694) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 21.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742706) |

- **stage-b-test-2-npu-a3**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，gh-proxy.test.osinfra.cn代理返回418错误，无法获取文件，属于网络或代理环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742297

- **stage-b-test-8-npu-a3**: 在下载custom-ops-2026.7.27-torch2.10.0-cann9.0.0-a3-aarch64.zip时，代理服务器返回418错误，可能是代理限制或文件不存在，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742299

- **stage-b-test-4-npu-a3 (1)**: 在下载ops-transformer zip包时，gh-proxy.test.osinfra.cn代理返回418错误，导致下载失败，作业退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742313

- **stage-b-test-1-npu-a3**: 作业在安装sgl-kernel-npu时，通过gh-proxy代理下载zip包，服务器返回418错误（疑似代理拒绝或限流），导致进程退出码1，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742320

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是上游产物未上传或存储配置变更，需检查相关依赖文件是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742353

- **multimodal-gen-test-1-npu-a3**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是文件被清理、路径错误或上传失败，属于外部存储环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742358

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是日志上传或依赖下载路径错误，或存储被清理。需检查作业配置中的blob路径或重试。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742425

- **stage-b-test-4-npu-a3 (0)**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release包，但代理返回418错误（疑似被拒绝或限流），导致下载失败，进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742459

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在存储中不存在。可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742694

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是数据未上传或已被删除，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742706

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30975146994/job/92233742397) |


## [Run #30974742424](https://github.com/sgl-project/sglang/actions/runs/30974742424)
- **分支**: `codex/deepgemm-memory-aware-layout`
- **总耗时**: 204.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30974742424

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 2.1min | 环境问题 | 下载 custom-ops 压缩包时 HTTP 418 错误，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396655) |
| stage-b-test-16-npu-a3 | 2.3min | 环境问题 | 下载sgl-kernel-npu依赖时代理返回418错误，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396667) |
| multimodal-gen-test-2-npu-a3 | 24.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396750) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 5.9min | 环境问题 | 作业在启动后立即失败，未执行实际测试，可能因环境或资源问题导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206397139) |

- **stage-b-test-8-npu-a3**: 在下载 sgl-kernel-npu 的 custom-ops-2026.7.27 包时，代理服务器返回 418 错误，可能是代理拒绝或资源不可用，属于外部依赖下载失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396655

- **stage-b-test-16-npu-a3**: 作业在安装sgl-kernel-npu时，通过gh-proxy.test.osinfra.cn代理下载GitHub release文件，但代理返回HTTP 418错误，下载失败，导致进程退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396667

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js版本弃用警告和artifact上传提示（无文件）。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396750

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示作业在准备阶段后直接进入清理，未运行测试用例，且无错误信息，可能因NPU资源分配失败或环境初始化异常导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206397139

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396647) |
| stage-b-test-1-npu-a3 | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396657) |
| stage-b-test-2-npu-a3 | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396672) |
| stage-b-test-4-npu-a3 (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396682) |
| stage-b-test-4-npu-a3 (1) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396717) |
| multimodal-gen-test-1-npu-a3 | 27.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206396759) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974742424/job/92206397138) |


## [Run #30974500284](https://github.com/sgl-project/sglang/actions/runs/30974500284)
- **分支**: `dev/fanshuaishuai/feat_overlap_image_load`
- **总耗时**: 207.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30974500284

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 2.9min | 环境问题 | 下载依赖包时代理返回418错误 | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679465) |
| multimodal-gen-test-2-npu-a3 | 25.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679509) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 3.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679793) |

- **stage-b-test-16-npu-a3**: 从gh-proxy.test.osinfra.cn下载custom-ops zip包时，代理服务器返回HTTP 418错误，导致下载失败，作业退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679465

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679509

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含runner初始化、依赖下载和作业后清理步骤，未展示测试执行过程及错误输出，无法定位具体失败原因，可能为日志采集不完整或作业在启动阶段即异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679793

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 8.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679451) |
| stage-b-test-2-npu-a3 | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679463) |
| stage-a-unit-test-npu | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679466) |
| stage-b-test-1-npu-a3 | 26.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679481) |
| stage-b-test-4-npu-a3 (0) | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679491) |
| stage-b-test-4-npu-a3 (1) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679494) |
| multimodal-gen-test-1-npu-a3 | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679526) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30974500284/job/92205679765) |


---
*Auto-generated by npu_pr_monitor.py*