# NPU CI 执行监控
**生成时间**: 2026-08-05 00:07 UTC
**分析 Run 数**: 10

---

## [Run #30959115359](https://github.com/sgl-project/sglang/actions/runs/30959115359)
- **分支**: `codex/deepgemm-memory-aware-layout`
- **总耗时**: 22.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30959115359

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 21.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938647) |
| stage-b-test-16-npu-a3 | 21.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938649) |
| stage-b-test-2-npu-a3 | 17.9min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938674) |
| multimodal-gen-test-2-npu-a3 | 21.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938690) |
| stage-b-test-4-npu-a3 (0) | 21.7min | 环境问题 | 自定义容器执行失败，NPU作业在初始化阶段崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938708) |
| stage-b-test-1-npu-a3 | 19.2min | 环境问题 | 自定义容器执行失败，NPU后端算子回退导致服务异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938737) |
| multimodal-gen-test-1-npu-a3 | 21.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938743) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 21.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938980) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 21.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938995) |

- **stage-b-test-8-npu-a3**: 作业 stage-b-test-8-npu-a3 在尝试下载日志或测试数据时，Azure Blob 返回 BlobNotFound 错误，说明对应 blob 已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938647

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938649

- **stage-b-test-2-npu-a3**: 作业在启动TokenizerManager后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938674

- **multimodal-gen-test-2-npu-a3**: 作业在访问Azure Blob存储时遇到BlobNotFound错误，说明所需的数据文件或资源未上传或已被删除，属于环境或资源准备问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938690

- **stage-b-test-4-npu-a3 (0)**: 日志显示TP0-TP3初始化正常，但随后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938708

- **stage-b-test-1-npu-a3**: 日志显示NPU后端不支持aten::_assert_async算子，回退到CPU执行，随后自定义容器实现执行失败，导致作业终止。属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938737

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938743

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938980

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程数据文件缺失或路径错误，可能是数据未上传或已被删除，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938995

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938738) |
| stage-b-test-4-npu-a3 (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30959115359/job/92158938739) |


## [Run #30958183786](https://github.com/sgl-project/sglang/actions/runs/30958183786)
- **分支**: `dev/fanshuaishuai/feat_overlap_image_load`
- **总耗时**: 24.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30958183786

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 23.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052309) |
| multimodal-gen-test-2-npu-a3 | 23.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052318) |
| stage-b-test-1-npu-a3 | 8.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052326) |
| stage-b-test-8-npu-a3 | 5.6min | 环境问题 | NPU服务启动后健康检查持续返回503，容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052328) |
| stage-b-test-4-npu-a3 (0) | 23.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052352) |
| multimodal-gen-test-1-npu-a3 | 23.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052367) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 23.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052721) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.7min | 环境问题 | Azure Blob 存储中指定的模型权重文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052837) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052309

- **multimodal-gen-test-2-npu-a3**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052318

- **stage-b-test-1-npu-a3**: 作业在运行NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员。日志显示torch_npu配置警告后容器中断，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052326

- **stage-b-test-8-npu-a3**: 服务启动后/health_generate接口持续503，说明模型未就绪或初始化失败，最终自定义容器执行失败，属于NPU环境或启动配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052328

- **stage-b-test-4-npu-a3 (0)**: 日志显示测试服务正常启动并处理请求，但随后出现'Executing the custom container implementation failed'错误，属于自托管runner容器环境问题，非代码或测试逻辑失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052352

- **multimodal-gen-test-1-npu-a3**: 作业在访问Azure Blob存储时遇到BlobNotFound错误，可能是由于文件被删除、路径错误或存储账户配置问题，导致CI无法继续执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052367

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052721

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明模型权重（glm5_top64_pruned_bf16）在存储中缺失或路径错误，可能是上传失败或配置错误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052837

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052322) |
| stage-b-test-2-npu-a3 | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052324) |
| stage-b-test-4-npu-a3 (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30958183786/job/92156052350) |


## [Run #30957291355](https://github.com/sgl-project/sglang/actions/runs/30957291355)
- **分支**: `main`
- **总耗时**: 10.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30957291355

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 9.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153297980) |
| multimodal-gen-test-1-npu-a3 | 9.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153297995) |
| stage-b-test-16-npu-a3 | 9.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298010) |
| stage-b-test-8-npu-a3 | 5.7min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298011) |
| stage-b-test-1-npu-a3 | 8.9min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298025) |
| stage-b-test-4-npu-a3 (1) | 7.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298040) |
| multimodal-gen-test-2-npu-a3 | 9.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298056) |
| stage-b-test-4-npu-a3 (0) | 1.9min | 环境问题 | 自定义容器执行失败，依赖安装过程中被中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298069) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 9.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298398) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298433) |

- **stage-b-test-2-npu-a3**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于基础设施/容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153297980

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153297995

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298010

- **stage-b-test-8-npu-a3**: 测试刚开始执行test_gsm8k时，自定义容器实现执行失败，导致作业终止。可能是容器环境或资源问题，非测试本身逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298011

- **stage-b-test-1-npu-a3**: 日志显示测试运行正常，但在22:50:45时出现错误："Executing the custom container implementation failed. Please contact your self hosted runner administrator."，表明是runner环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298025

- **stage-b-test-4-npu-a3 (1)**: 日志显示测试运行中服务正常响应，但随后报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298040

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298056

- **stage-b-test-4-npu-a3 (0)**: 作业在安装triton-ascend等依赖时，卸载旧版本包后，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298069

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储资源缺失或路径错误，可能是文件被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298398

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或数据文件未上传或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153298433

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30957291355/job/92153297999) |


## [Run #30957205516](https://github.com/sgl-project/sglang/actions/runs/30957205516)
- **分支**: `qiaolin_fused_commit_indices`
- **总耗时**: 31.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30957205516

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 3.0min | 环境问题 | 自托管runner执行自定义容器实现失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022256) |
| stage-b-test-1-npu-a3 | 25.6min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022267) |
| multimodal-gen-test-2-npu-a3 | 25.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022319) |
| stage-b-test-4-npu-a3 (0) | 25.6min | 环境问题 | NPU测试中服务健康检查返回503，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022324) |
| multimodal-gen-test-1-npu-a3 | 25.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022346) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 9.2min | 其他 | 日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022685) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 25.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022714) |

- **stage-b-test-16-npu-a3**: 作业在安装triton-ascend依赖时，卸载numpy过程中自定义容器实现报错，导致执行中断，属于runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022256

- **stage-b-test-1-npu-a3**: 作业在模型加载和KV Cache分配完成后，执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于NPU环境或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022267

- **multimodal-gen-test-2-npu-a3**: 作业在尝试访问Azure Blob存储时，遇到BlobNotFound错误，说明所需的数据文件或资源未正确上传或已被删除，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022319

- **stage-b-test-4-npu-a3 (0)**: 日志显示服务启动后/health_generate接口持续返回503，且存在NPU算子回退警告，最终容器执行失败，属于环境或服务初始化问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022324

- **multimodal-gen-test-1-npu-a3**: 作业日志返回BlobNotFound错误，说明CI流程尝试访问的存储资源缺失或路径错误，属于环境配置或资源准备问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022346

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志在测试开始前即结束，未包含实际测试命令或错误信息，无法判断失败原因，可能为日志截断或作业被提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022685

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在指定容器中不存在，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022714

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022269) |
| stage-b-test-8-npu-a3 | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022289) |
| stage-b-test-4-npu-a3 (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022304) |
| stage-b-test-2-npu-a3 | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30957205516/job/92153022308) |


## [Run #30955086858](https://github.com/sgl-project/sglang/actions/runs/30955086858)
- **分支**: `main`
- **总耗时**: 6.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30955086858

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a3 | 3.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294171) |
| stage-b-test-16-npu-a3 | 0.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294201) |
| stage-b-test-1-npu-a3 | 5.4min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294225) |
| stage-b-test-4-npu-a3 (1) | 5.5min | 环境问题 | NPU容器在加载模型权重后执行自定义容器实现时崩溃，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294238) |
| stage-b-test-8-npu-a3 | 3.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294240) |
| stage-b-test-4-npu-a3 (0) | 5.5min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294241) |
| multimodal-gen-test-2-npu-a3 | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294270) |
| multimodal-gen-test-1-npu-a3 | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294299) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294850) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 2.4min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294921) |

- **stage-b-test-2-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误提示联系自托管 runner 管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294171

- **stage-b-test-16-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于基础设施或容器配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294201

- **stage-b-test-1-npu-a3**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294225

- **stage-b-test-4-npu-a3 (1)**: 日志显示模型权重加载成功（7/7 shards），但在获取ASCEND_OPP_PATH环境变量后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器运行时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294238

- **stage-b-test-8-npu-a3**: 作业在运行测试命令时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是K8s容器调度或环境配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294240

- **stage-b-test-4-npu-a3 (0)**: 日志显示在捕获批次过程中，可用内存逐渐下降（8.71GB降至8.65GB），随后报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294241

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294270

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294299

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294850

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 提供的日志仅包含作业启动、环境准备和清理信息，中间部分被省略，未包含测试执行、断言或错误输出，因此无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294921

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30955086858/job/92146294246) |


## [Run #30953860922](https://github.com/sgl-project/sglang/actions/runs/30953860922)
- **分支**: `main`
- **总耗时**: 18.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30953860922

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-8-npu-a3 | 16.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502051) |
| stage-b-test-4-npu-a3 (0) | 7.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502063) |
| stage-b-test-2-npu-a3 | 16.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502068) |
| stage-b-test-16-npu-a3 | 16.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502073) |
| stage-b-test-4-npu-a3 (1) | 9.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502089) |
| multimodal-gen-test-1-npu-a3 | 16.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502122) |
| multimodal-gen-test-2-npu-a3 | 16.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502124) |
| stage-b-test-1-npu-a3 | 9.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502196) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502486) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 7.5min | 其他 | 日志被截断，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502513) |

- **stage-b-test-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502051

- **stage-b-test-4-npu-a3 (0)**: 日志显示测试运行正常，但在22:06:17时出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502063

- **stage-b-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502068

- **stage-b-test-16-npu-a3**: 作业 stage-b-test-16-npu-a3 在尝试下载或访问 Azure Blob 中的日志时，返回 BlobNotFound 错误（HTTP 404）。可能是日志文件被删除、路径错误或上传未完成，属于外部存储依赖问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502073

- **stage-b-test-4-npu-a3 (1)**: 日志显示服务启动后，自定义容器实现执行失败（Executing the custom container implementation failed），可能是NPU资源或容器环境问题，导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502089

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502122

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问某个blob时，返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置问题，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502124

- **stage-b-test-1-npu-a3**: 作业在运行NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员。日志显示torch_npu相关警告，但无明确测试失败信息，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502196

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或测试数据未上传到指定存储路径，或路径配置错误。需检查相关 blob 是否存在及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502486

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业启动和清理信息，未显示测试执行过程或错误输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502513

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30953860922/job/92142502125) |


## [Run #30948603482](https://github.com/sgl-project/sglang/actions/runs/30948603482)
- **分支**: `feat/grpc-generation-controls`
- **总耗时**: 190.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30948603482

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847093) |
| stage-b-test-16-npu-a3 | 10.0min | 环境问题 | NPU PD分离测试失败，测试用例执行报错 | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847165) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 25.2min | 其他 | 日志未显示测试失败原因，仅包含作业启动和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847753) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847093

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py测试在NPU A3环境下运行352秒后失败，返回错误码1，测试摘要显示0/6通过，可能是环境配置或依赖问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847165

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志截断，未包含测试执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847753

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124846985) |
| stage-b-test-1-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124846997) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847002) |
| multimodal-gen-test-1-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847047) |
| stage-b-test-4-npu-a3 (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847074) |
| stage-b-test-4-npu-a3 (1) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847078) |
| stage-b-test-2-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847090) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948603482/job/92124847719) |


## [Run #30948228847](https://github.com/sgl-project/sglang/actions/runs/30948228847)
- **分支**: `cheng/unified-memory-pd-disagg`
- **总耗时**: 197.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30948228847

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.4min | 其他 | 作业未显示明确失败原因，仅上传工件时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669195) |
| stage-b-test-16-npu-a3 | 9.3min | 超时 | NPU PD disaggregation 测试超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669212) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时无文件，可能测试通过但未生成失败产物，或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669195

- **stage-b-test-16-npu-a3**: 测试 test_npu_pd_disaggregation.py 运行 348 秒后失败，超过预估的 400 秒限制，导致 0/6 测试通过，作业退出码 255。
  链接: https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669212

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669176) |
| stage-b-test-4-npu-a3 (0) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669187) |
| stage-b-test-2-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669206) |
| stage-b-test-1-npu-a3 | 26.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669219) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669270) |
| stage-b-test-4-npu-a3 (1) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669407) |
| multimodal-gen-test-1-npu-a3 | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669413) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669845) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30948228847/job/92123669849) |


## [Run #30947612300](https://github.com/sgl-project/sglang/actions/runs/30947612300)
- **分支**: `main`
- **总耗时**: 47.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30947612300

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 30.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509093) |
| multimodal-gen-test-1-npu-a3 | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509098) |
| stage-b-test-4-npu-a3 (0) | 37.5min | 代码错误 | NPU DP注意力测试失败，测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509114) |
| stage-b-test-1-npu-a3 | 38.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509127) |
| multimodal-gen-test-2-npu-a3 | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509146) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 46.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509879) |

- **stage-b-test-16-npu-a3**: 作业在启动NPU服务时，Watchdog TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509093

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509098

- **stage-b-test-4-npu-a3 (0)**: test_npu_dp_attention.py测试失败，退出码1，耗时1576秒远超预估400秒，可能因超时或断言失败导致，需检查该测试用例具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509114

- **stage-b-test-1-npu-a3**: 日志显示服务正常运行，但GitHub Actions在运行自定义容器实现时失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509127

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509146

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509879

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-8-npu-a3 | 12.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509091) |
| stage-b-test-4-npu-a3 (1) | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509112) |
| stage-b-test-2-npu-a3 | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509170) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509182) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947612300/job/92121509782) |


## [Run #30947593692](https://github.com/sgl-project/sglang/actions/runs/30947593692)
- **分支**: `mxz/rope-cache-rebuild-dead-buffers`
- **总耗时**: 34.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30947593692

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a3 | 22.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762370) |
| stage-b-test-16-npu-a3 | 12.7min | 超时 | NPU PD分离测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762389) |
| multimodal-gen-test-2-npu-a3 | 33.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762400) |
| multimodal-gen-test-1-npu-a3 | 33.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762473) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 33.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762799) |
| single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k | 23.0min | 其他 | 日志被截断，未显示实际测试结果，无法判断失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762827) |

- **stage-b-test-1-npu-a3**: 测试运行到第9个用例时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762370

- **stage-b-test-16-npu-a3**: test_npu_pd_disaggregation.py 运行465秒，超过预估的400秒，最终超时失败，0/6测试通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762389

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure 存储中缺失，可能是资源未上传或路径错误，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762400

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762473

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-800t-2, test/registered/n... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在存储中缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762799

- **single-node-poc (glm5_top64_pruned_bf16_8p_gsm8k, linux-aarch64-a3-16-, test/registered/npu/accur... / glm5_top64_pruned_bf16_8p_gsm8k**: 日志仅包含作业启动、环境准备和清理信息，未包含测试执行过程或错误输出，可能因日志截断或作业在测试前被取消，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762827

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a3 | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762334) |
| stage-b-test-4-npu-a3 (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762353) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762363) |
| stage-b-test-8-npu-a3 | 8.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762393) |
| stage-b-test-4-npu-a3 (0) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30947593692/job/92146762436) |


---
*Auto-generated by npu_pr_monitor.py*