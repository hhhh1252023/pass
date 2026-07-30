# NPU CI 执行监控
**生成时间**: 2026-07-30 12:08 UTC
**分析 Run 数**: 26

---

## [Run #30538507758](https://github.com/sgl-project/sglang/actions/runs/30538507758)
- **分支**: `tom/revert-pr10414`
- **总耗时**: 12.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30538507758

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 11.9min | 环境问题 | 自定义容器执行失败，可能是NPU环境或资源问题 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531280) |
| stage-b-test-16-npu-a3 | 11.9min | 环境问题 | 自定义容器执行失败，可能是NPU环境或资源问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531289) |
| stage-b-test-1-npu-a2 (0) | 11.9min | 环境问题 | 自定义容器执行失败，可能是NPU环境或资源问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531312) |
| stage-b-test-1-npu-a2 (1) | 11.9min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531317) |
| multimodal-gen-test-2-npu-a3 | 12.0min | 其他 | 作业日志中未显示测试失败或错误信息，仅包含Node.js版本弃用警告和工件上传提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531319) |
| stage-b-test-2-npu-a2 (0) | 5.6min | 环境问题 | 自定义容器执行失败，可能是自托管运行器环境配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531333) |
| multimodal-gen-test-1-npu-a3 | 12.0min | 其他 | 作业日志不完整，未显示测试执行与失败信息，仅包含环境准备和清理步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531349) |
| stage-b-test-2-npu-a2 (1) | 5.4min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531369) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 12.0min | 其他 | 日志未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531792) |

- **stage-b-test-4-npu-a3**: 日志显示在加载模型权重时出现'Executing the custom container implementation failed'错误，提示联系自托管运行器管理员，表明NPU环境或容器配置存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531280

- **stage-b-test-16-npu-a3**: 日志显示在加载shards过程中出现错误："Executing the custom container implementation failed"，建议联系自托管运行器管理员检查NPU环境配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531289

- **stage-b-test-1-npu-a2 (0)**: 日志显示`Executing the custom container implementation failed`，提示联系自托管运行器管理员，表明NPU容器环境异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531312

- **stage-b-test-1-npu-a2 (1)**: 日志显示 'Executing the custom container implementation failed'，提示联系自托管运行器管理员，表明是运行器环境或容器配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531317

- **multimodal-gen-test-2-npu-a3**: 日志仅包含Node.js 20弃用警告、工件上传时未找到文件（diffusion-failures/目录为空）等非致命信息，未出现测试失败、超时或环境错误等明确失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531319

- **stage-b-test-2-npu-a2 (0)**: 日志显示 `Executing the custom container implementation failed`，提示联系自托管运行器管理员，表明是运行器环境或容器配置异常导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531333

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions runner初始化、Node版本弃用警告及artifact上传（未找到文件），未出现任何测试用例运行或失败的具体输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531349

- **stage-b-test-2-npu-a2 (1)**: 日志显示`Executing the custom container implementation failed`，表明运行自定义容器时出错，且存在Node.js版本警告，但核心失败原因是容器执行异常，需联系管理员检查自托管Runner环境。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531369

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志截断，缺少测试执行和失败的关键输出，无法判断具体失败原因。可能为测试未运行或日志不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531792

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30538507758/job/90857531279) |


## [Run #30537382613](https://github.com/sgl-project/sglang/actions/runs/30537382613)
- **分支**: `agent/remove-orphan-aot-headers`
- **总耗时**: 43.6min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30537382613

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30537382613/job/90853891571) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30537382613/job/90853891592) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30537382613/job/90853891624) |
| stage-b-test-2-npu-a2 (1) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30537382613/job/90853891625) |
| stage-a-unit-test-npu | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30537382613/job/90853891630) |
| stage-b-test-1-npu-a2 (0) | 43.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30537382613/job/90853891654) |
| stage-b-test-4-npu-a3 | 37.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30537382613/job/90853891655) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30537382613/job/90853891898) |


## [Run #30534703847](https://github.com/sgl-project/sglang/actions/runs/30534703847)
- **分支**: `perf/hisparse-eliminate-redundant-fill`
- **总耗时**: 52.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30534703847

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 33.4min | 其他 | 作业日志不完整，未显示测试失败的具体错误信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190796) |

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备、Node版本警告和上传artifact步骤，缺少测试执行阶段的输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190796

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190708) |
| stage-b-test-16-npu-a3 | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190732) |
| stage-b-test-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190743) |
| stage-b-test-1-npu-a2 (0) | 41.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190749) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190769) |
| stage-b-test-2-npu-a2 (1) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190795) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190818) |
| multimodal-gen-test-1-npu-a3 | 26.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845190828) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30534703847/job/90845191339) |


## [Run #30533952171](https://github.com/sgl-project/sglang/actions/runs/30533952171)
- **分支**: `main`
- **总耗时**: 59.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30533952171

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882806) |
| multimodal-gen-test-1-npu-a3 | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882811) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882832) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882866) |
| stage-b-test-2-npu-a2 (1) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882868) |
| stage-b-test-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882881) |
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882892) |
| multimodal-gen-test-2-npu-a3 | 43.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882909) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842882919) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533952171/job/90842883305) |


## [Run #30533172394](https://github.com/sgl-project/sglang/actions/runs/30533172394)
- **分支**: `main`
- **总耗时**: 58.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30533172394

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 15.9min | 代码错误 | HiCache MLA测试失败，返回非零退出码 | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170740) |
| stage-b-test-1-npu-a2 (0) | 8.7min | 代码错误 | HiCache MHA测试失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170761) |

- **stage-b-test-4-npu-a3**: 测试文件test_npu_hicache_mla.py执行失败（exit code 1），5个测试中仅1个通过，具体错误需查看该测试日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170740

- **stage-b-test-1-npu-a2 (0)**: 测试`test_npu_hicache_mha.py`执行失败，0/5测试通过，具体错误原因需查看该测试脚本内部日志，可能与HiCache功能或配置有关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170761

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170719) |
| stage-b-test-2-npu-a2 (0) | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170769) |
| stage-a-unit-test-npu | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170780) |
| stage-b-test-1-npu-a2 (1) | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170781) |
| stage-b-test-16-npu-a3 | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170825) |
| multimodal-gen-test-1-npu-a3 | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170827) |
| multimodal-gen-test-2-npu-a3 | 46.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840170849) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30533172394/job/90840171295) |


## [Run #30532985462](https://github.com/sgl-project/sglang/actions/runs/30532985462)
- **分支**: `mick/encoder-parallel-unified`
- **总耗时**: 66.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30532985462

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 51.3min | 其他 | 日志未显示明确失败原因，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532985462/job/90839580424) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾无错误或退出码，仅显示Node 20弃用警告及工件上传成功，作业可能因外部中断或超时被终止，但日志不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532985462/job/90839580424

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 35.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30532985462/job/90839580416) |


## [Run #30532682725](https://github.com/sgl-project/sglang/actions/runs/30532682725)
- **分支**: `main`
- **总耗时**: 19.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30532682725

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 18.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559363) |
| stage-b-test-16-npu-a3 | 18.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559366) |
| stage-b-test-1-npu-a2 (0) | 3.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559396) |
| stage-b-test-1-npu-a2 (1) | 3.4min | 环境问题 | 自定义容器实现执行失败，可能是自托管运行器环境配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559417) |
| stage-a-unit-test-npu | 4.1min | 环境问题 | 自定义容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559419) |
| stage-b-test-4-npu-a3 | 5.2min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559431) |
| multimodal-gen-test-1-npu-a3 | 18.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559444) |
| stage-b-test-2-npu-a2 (0) | 18.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559470) |
| stage-b-test-2-npu-a2 (1) | 0.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559472) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 18.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致模型权重或数据下载失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559804) |

- **multimodal-gen-test-2-npu-a3**: 作业尝试访问 Azure Blob 存储中的某个 blob，但该 blob 已被删除或路径错误，返回 BlobNotFound 错误。需检查存储路径或 blob 是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559363

- **stage-b-test-16-npu-a3**: CI 作业尝试下载一个不存在的 blob 文件，导致失败。可能是构建产物未正确上传或路径错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559366

- **stage-b-test-1-npu-a2 (0)**: 在下载 torch_npu-2.10.0 时，自定义容器实现执行失败，提示请联系自托管运行器管理员，可能是容器环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559396

- **stage-b-test-1-npu-a2 (1)**: 日志显示在解压自定义容器后出现 'Executing the custom container implementation failed' 错误，提示联系自托管运行器管理员，表明是运行器环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559417

- **stage-a-unit-test-npu**: 日志显示在安装依赖后，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，建议联系自托管运行器管理员。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559419

- **stage-b-test-4-npu-a3**: 日志显示服务器启动成功，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管运行器管理员，表明是容器环境问题而非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559431

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，表明 CI 作业尝试访问的 Azure Blob 存储资源已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559444

- **stage-b-test-2-npu-a2 (0)**: 日志返回 BlobNotFound 错误，表明 CI 依赖的某个 blob 资源（如模型权重或数据文件）已被删除或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559470

- **stage-b-test-2-npu-a2 (1)**: 日志显示在执行自定义容器实现时出错，错误信息为“Executing the custom container implementation failed”，建议联系自托管运行器管理员排查容器配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559472

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业尝试从 Azure Blob 下载资源时返回 BlobNotFound 错误，可能是模型权重文件路径错误、存储容器配置变更或资源被删除，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30532682725/job/90838559804


## [Run #30531835261](https://github.com/sgl-project/sglang/actions/runs/30531835261)
- **分支**: `main`
- **总耗时**: 12.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30531835261

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844070) |
| stage-b-test-2-npu-a2 (1) | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844136) |
| stage-b-test-1-npu-a2 (1) | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844139) |
| multimodal-gen-test-1-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844142) |
| stage-b-test-2-npu-a2 (0) | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844148) |
| stage-b-test-4-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844149) |
| stage-b-test-16-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844171) |
| stage-b-test-1-npu-a2 (0) | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844185) |
| multimodal-gen-test-2-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844237) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844718) |

- **stage-a-unit-test-npu**: 作业尝试访问 Azure Blob 存储中的某个 blob，但该 blob 已被删除或路径错误，返回 BlobNotFound 错误。可能是依赖的预构建工件或缓存丢失。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844070

- **stage-b-test-2-npu-a2 (1)**: 日志返回 BlobNotFound 错误，表明 CI 依赖的某个 blob 文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844136

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，表明 CI 依赖的某个文件或资源在 Azure 存储中缺失，可能是由于清理、路径错误或上传失败导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844139

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，表明 CI 作业尝试访问的 Azure Blob 存储资源已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844142

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure 存储中缺失，可能是构建产物或数据未正确上传或已被删除。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844148

- **stage-b-test-4-npu-a3**: 日志返回 BlobNotFound 错误，表明 CI 依赖的某个 blob 资源（如模型权重或数据文件）已被删除或路径错误，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844149

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，表明 CI 作业尝试访问的 Azure Blob 存储资源已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844171

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，表明 CI 依赖的某个文件或资源在 Azure 存储中缺失，可能是由于存储清理、路径错误或上传失败导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844185

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，表明 CI 作业尝试访问的 Azure Blob 存储资源已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844237

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业尝试访问 Azure Blob 存储中的某个 blob，但该 blob 已被删除或路径错误，返回 BlobNotFound 错误。这属于外部依赖资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531835261/job/90835844718


## [Run #30531399613](https://github.com/sgl-project/sglang/actions/runs/30531399613)
- **分支**: `main`
- **总耗时**: 6.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30531399613

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478849) |
| multimodal-gen-test-1-npu-a3 | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478852) |
| stage-b-test-1-npu-a2 (0) | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478862) |
| stage-b-test-4-npu-a3 | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478865) |
| stage-b-test-2-npu-a2 (0) | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478876) |
| multimodal-gen-test-2-npu-a3 | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478895) |
| stage-b-test-1-npu-a2 (1) | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478899) |
| stage-b-test-16-npu-a3 | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478921) |
| stage-b-test-2-npu-a2 (1) | 5.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478977) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.9min | 环境问题 | 依赖的blob文件不存在导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834479430) |

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，表明 CI 尝试访问的 Azure 存储 blob 已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478849

- **multimodal-gen-test-1-npu-a3**: 作业尝试访问 Azure Blob 存储中的某个 blob 文件，但该文件不存在（BlobNotFound），导致任务失败。可能是依赖的模型权重或数据文件未正确上传或路径错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478852

- **stage-b-test-1-npu-a2 (0)**: 日志显示 BlobNotFound 错误，表明 CI 作业尝试访问的 Azure Blob 存储资源已被删除或路径错误，导致无法获取所需文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478862

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，表明 CI 作业尝试访问的 Azure Blob 存储资源已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478865

- **stage-b-test-2-npu-a2 (0)**: 日志显示 BlobNotFound 错误，表明 CI 尝试访问的 Azure Blob 存储资源已被删除或路径错误，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478876

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，表明 CI 作业尝试访问的 Azure Blob 存储资源已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478895

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，表明 CI 依赖的某个文件或资源在 Azure 存储中缺失，可能是由于清理、路径错误或上传失败导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478899

- **stage-b-test-16-npu-a3**: 作业尝试访问 Azure Blob 存储中的某个 blob，但该 blob 已被删除或路径错误，返回 BlobNotFound 错误。这属于外部依赖或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478921

- **stage-b-test-2-npu-a2 (1)**: 作业尝试访问 Azure Blob 存储中的某个 blob 文件，但该文件不存在（BlobNotFound），可能是依赖的预构建工件或数据文件缺失或路径错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834478977

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示Azure Blob返回BlobNotFound错误，说明作业所需的模型权重或数据文件在存储中缺失，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531399613/job/90834479430


## [Run #30531338800](https://github.com/sgl-project/sglang/actions/runs/30531338800)
- **分支**: `mick/encoder-parallel-unified`
- **总耗时**: 24.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30531338800

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 23.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531338800/job/90834342936) |
| multimodal-gen-test-2-npu-a3 | 23.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531338800/job/90834342971) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，表明 CI 依赖的某个 blob 文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531338800/job/90834342936

- **multimodal-gen-test-2-npu-a3**: 作业尝试访问 Azure Blob 存储中的某个 blob 文件，但该文件不存在（BlobNotFound），可能是依赖的模型或数据文件缺失或路径错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531338800/job/90834342971


## [Run #30531240658](https://github.com/sgl-project/sglang/actions/runs/30531240658)
- **分支**: `add-pr-tests`
- **总耗时**: 65.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30531240658

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.2min | 其他 | 作业未执行测试逻辑，仅上传了不存在的失败文件路径。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923671) |
| stage-b-test-1-npu-a2 (0) | 30.1min | 环境问题 | 自定义容器执行失败，可能是NPU资源或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923718) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 28.8min | 其他 | 日志不完整，未显示测试执行结果或错误信息 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924193) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.3min | 其他 | 日志不完整，未显示测试执行和失败的具体错误信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924236) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 9.7min | 其他 | 日志不完整，未显示测试执行和失败的具体错误信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924249) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 9.7min | 其他 | 日志不完整，未显示测试执行和失败信息 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924260) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 9.7min | 其他 | 日志不完整，未显示测试执行和失败原因 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924273) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 9.7min | 其他 | 日志不完整，未显示测试执行结果，仅包含CI基础设施信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924301) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业仅执行了checkout和upload-artifact，未找到diffusion-failures/目录，无实际测试运行，可能因前置步骤失败或配置错误导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923671

- **stage-b-test-1-npu-a2 (0)**: 作业在捕获NPU图时失败，错误信息为自定义容器实现执行失败，建议联系自托管运行器管理员检查NPU环境或容器配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923718

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志仅包含CI环境初始化、Node.js版本警告和清理步骤，未提供测试用例的实际运行输出或失败堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924193

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志仅包含CI环境准备和清理步骤，缺少测试运行阶段的输出，无法判断失败原因。可能是日志截断或作业在测试执行前已失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924236

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 提供的日志仅包含CI环境初始化和清理步骤，缺少测试运行阶段的输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924249

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 提供的日志仅包含作业初始化、环境准备和清理步骤，缺少测试执行阶段的输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924260

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 提供的日志仅包含CI作业的初始化和清理阶段，缺少测试执行的关键输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924273

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志在测试执行前中断，仅包含runner初始化、依赖下载和清理步骤，未提供任何测试失败的具体错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833924301

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 11.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923694) |
| stage-b-test-2-npu-a2 (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923704) |
| stage-b-test-2-npu-a2 (1) | 11.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923716) |
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923723) |
| stage-b-test-4-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923734) |
| multimodal-gen-test-2-npu-a3 | 36.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923735) |
| stage-b-test-1-npu-a2 (1) | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30531240658/job/90833923794) |


## [Run #30530667574](https://github.com/sgl-project/sglang/actions/runs/30530667574)
- **分支**: `feat/llada2-block-routing`
- **总耗时**: 70.7min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30530667574

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 37.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093139) |
| stage-b-test-4-npu-a3 | 37.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093147) |
| multimodal-gen-test-1-npu-a3 | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093169) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093206) |
| stage-b-test-2-npu-a2 (1) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093224) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093234) |
| stage-b-test-16-npu-a3 | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093248) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093322) |
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093383) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30530667574/job/90832093750) |


## [Run #30530523605](https://github.com/sgl-project/sglang/actions/runs/30530523605)
- **分支**: `mick/encoder-parallel-unified`
- **总耗时**: 12.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30530523605

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 2.4min | 环境问题 | Node.js 20 被弃用，但工作流仍强制使用，导致兼容性警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30530523605/job/90831713079) |
| multimodal-gen-test-2-npu-a3 | 3.6min | 其他 | 作业日志不完整，未显示测试执行和失败信息，仅包含环境准备和清理步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30530523605/job/90831713178) |

- **multimodal-gen-test-1-npu-a3**: GitHub Actions 运行器默认使用 Node 24，但 actions/checkout 和 upload-artifact 仍基于 Node 20，触发弃用警告，可能影响后续步骤稳定性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30530523605/job/90831713079

- **multimodal-gen-test-2-npu-a3**: 日志中只有GitHub Actions的初始化、Node版本警告和上传工件步骤，缺少实际测试命令的输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30530523605/job/90831713178


## [Run #30529598756](https://github.com/sgl-project/sglang/actions/runs/30529598756)
- **分支**: `tom/revert-pr10414`
- **总耗时**: 65.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30529598756

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 33.9min | 代码错误 | 测试用例 test_npu_autoround_moe.py 执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594379) |

- **stage-b-test-1-npu-a2 (0)**: 在 NPU CI 的 stage-b-test-1-npu-a2 作业中，5个测试用例有3个通过，2个失败，其中 test_npu_autoround_moe.py 返回退出码1，导致作业整体失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594379

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594357) |
| stage-b-test-16-npu-a3 | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594358) |
| multimodal-gen-test-2-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594360) |
| stage-b-test-4-npu-a3 | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594362) |
| multimodal-gen-test-1-npu-a3 | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594369) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594398) |
| stage-b-test-2-npu-a2 (0) | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594423) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594473) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529598756/job/90828594685) |


## [Run #30529017362](https://github.com/sgl-project/sglang/actions/runs/30529017362)
- **分支**: `fix_mamba_l2_size`
- **总耗时**: 88.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30529017362

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116680) |
| multimodal-gen-test-2-npu-a3 | 52.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116691) |
| stage-b-test-16-npu-a3 | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116694) |
| stage-a-unit-test-npu | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116702) |
| stage-b-test-2-npu-a2 (0) | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116744) |
| stage-b-test-1-npu-a2 (1) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116752) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116753) |
| stage-b-test-1-npu-a2 (0) | 43.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116756) |
| stage-b-test-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834116778) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30529017362/job/90834117753) |


## [Run #30528890445](https://github.com/sgl-project/sglang/actions/runs/30528890445)
- **分支**: `fix/verify-splitkv-mla-guard`
- **总耗时**: 60.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528890445

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528890445/job/90826355481) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528890445/job/90826355485) |
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528890445/job/90826355504) |
| stage-b-test-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528890445/job/90826355517) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528890445/job/90826355579) |
| stage-b-test-16-npu-a3 | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528890445/job/90826355610) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528890445/job/90826355722) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528890445/job/90826355942) |


## [Run #30528717994](https://github.com/sgl-project/sglang/actions/runs/30528717994)
- **分支**: `bingxche/fix-dsv4-compact-moe-weight-load`
- **总耗时**: 67.2min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528717994

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751438) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751481) |
| stage-b-test-1-npu-a2 (1) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751502) |
| stage-a-unit-test-npu | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751503) |
| multimodal-gen-test-1-npu-a3 | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751519) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751523) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751524) |
| stage-b-test-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751536) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751620) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528717994/job/90825751853) |


## [Run #30528701434](https://github.com/sgl-project/sglang/actions/runs/30528701434)
- **分支**: `mick/encoder-parallel-unified`
- **总耗时**: 27.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528701434

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 1.2min | 其他 | 作业日志不完整，未显示测试执行和失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528701434/job/90825695469) |
| multimodal-gen-test-2-npu-a3 | 26.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528701434/job/90825695506) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions环境初始化、Node版本警告和清理步骤，未包含任何测试运行、编译或错误堆栈信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528701434/job/90825695469

- **multimodal-gen-test-2-npu-a3**: 作业尝试访问 Azure Blob 存储中的某个 blob，但该 blob 已被删除或路径错误，返回 BlobNotFound 错误。这属于外部依赖或配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528701434/job/90825695506


## [Run #30528667285](https://github.com/sgl-project/sglang/actions/runs/30528667285)
- **分支**: `mm_cache_abstract`
- **总耗时**: 68.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528667285

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 2.7min | 环境问题 | pip下载包时网络连接中断导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601037) |
| stage-b-test-2-npu-a2 (1) | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致 CI 作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601079) |

- **stage-b-test-2-npu-a2 (0)**: 在安装依赖时，pip下载过程中出现IncompleteRead错误，网络连接中断导致下载不完整，属于临时网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601037

- **stage-b-test-2-npu-a2 (1)**: 日志返回 BlobNotFound 错误，表明 CI 依赖的某个 blob 文件缺失或路径错误，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601079

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 9.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601001) |
| multimodal-gen-test-2-npu-a3 | 35.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601026) |
| stage-b-test-16-npu-a3 | 13.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601028) |
| multimodal-gen-test-1-npu-a3 | 34.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601036) |
| stage-b-test-4-npu-a3 | 39.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601048) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601050) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601103) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528667285/job/90825601489) |


## [Run #30528510320](https://github.com/sgl-project/sglang/actions/runs/30528510320)
- **分支**: `fix_eagle_shape`
- **总耗时**: 57.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528510320

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (1) | 3.6min | 环境问题 | pip下载依赖包时网络连接中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072268) |

- **stage-b-test-2-npu-a2 (1)**: 在安装Python依赖时，pip从PyPI下载包过程中发生IncompleteRead错误，仅读取了18MB但预期需要188MB，网络不稳定或代理问题导致连接中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072268

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072156) |
| multimodal-gen-test-1-npu-a3 | 37.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072172) |
| stage-b-test-16-npu-a3 | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072174) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072200) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072220) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072227) |
| stage-b-test-1-npu-a2 (1) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072229) |
| multimodal-gen-test-2-npu-a3 | 52.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072248) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528510320/job/90833072579) |


## [Run #30528477260](https://github.com/sgl-project/sglang/actions/runs/30528477260)
- **分支**: `feat/spectrum`
- **总耗时**: 36.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528477260

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528477260/job/90824992158) |
| multimodal-gen-test-2-npu-a3 | 36.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528477260/job/90824992203) |


## [Run #30528411309](https://github.com/sgl-project/sglang/actions/runs/30528411309)
- **分支**: `optimize-mem-pool-slot-alloc`
- **总耗时**: 45.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528411309

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769070) |
| stage-b-test-1-npu-a2 (0) | 45.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769074) |
| multimodal-gen-test-1-npu-a3 | 35.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769094) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769096) |
| stage-b-test-16-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769098) |
| stage-b-test-1-npu-a2 (1) | 33.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769108) |
| multimodal-gen-test-2-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769124) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769132) |
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769150) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528411309/job/90824769800) |


## [Run #30528263521](https://github.com/sgl-project/sglang/actions/runs/30528263521)
- **分支**: `fix-dsa-sparse-prefill-topk-length`
- **总耗时**: 43.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528263521

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 4.9min | 环境问题 | pip下载包时网络连接中断导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263702) |
| stage-b-test-1-npu-a2 (0) | 5.5min | 环境问题 | pip下载依赖包时网络连接中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263706) |
| stage-b-test-2-npu-a2 (0) | 5.2min | 环境问题 | pip下载包时网络连接中断导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263763) |
| stage-b-test-2-npu-a2 (1) | 5.3min | 环境问题 | 下载triton-ascend包时超时或失败，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263797) |

- **stage-b-test-1-npu-a2 (1)**: 在安装依赖时，pip下载包出现IncompleteRead错误，网络连接中断导致下载不完整，属于临时网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263702

- **stage-b-test-1-npu-a2 (0)**: 在安装Python依赖时，pip从远程下载包过程中发生IncompleteRead错误，网络连接不稳定导致下载不完整，最终安装脚本退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263706

- **stage-b-test-2-npu-a2 (0)**: 在安装依赖时，pip下载过程中出现IncompleteRead错误，网络连接中断，导致包下载不完整，最终CI流程失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263763

- **stage-b-test-2-npu-a2 (1)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl文件耗时过长，最终容器执行失败，可能是网络问题或包源不稳定。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263797

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263668) |
| stage-b-test-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263710) |
| multimodal-gen-test-1-npu-a3 | 36.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263713) |
| multimodal-gen-test-2-npu-a3 | 42.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263746) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824263761) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528263521/job/90824264178) |


## [Run #30528120300](https://github.com/sgl-project/sglang/actions/runs/30528120300)
- **分支**: `main`
- **总耗时**: 44.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30528120300

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 43.1min | 其他 | 作业日志显示所有测试通过，未发现失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823911986) |
| multimodal-gen-test-2-npu-a3 | 43.3min | 其他 | 日志未显示测试失败原因，仅包含环境警告和空工件上传。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823912046) |

- **stage-b-test-1-npu-a2 (0)**: 日志中Test Summary显示5/5 passed，所有测试均通过，无错误或失败信息。作业可能因其他未记录原因被标记为失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823911986

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅有Node.js版本弃用警告和上传空工件（diffusion-failures/目录无文件）的记录，无法判断实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823912046

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823911965) |
| multimodal-gen-test-1-npu-a3 | 38.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823911990) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823911995) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823911998) |
| stage-b-test-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823912003) |
| stage-b-test-16-npu-a3 | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823912026) |
| stage-b-test-2-npu-a2 (1) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823912058) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30528120300/job/90823912514) |


## [Run #30524819759](https://github.com/sgl-project/sglang/actions/runs/30524819759)
- **分支**: `mick/encoder-parallel-unified`
- **总耗时**: 43.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30524819759

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 43.2min | 其他 | 作业日志中未显示明确的失败错误，仅包含Node.js版本弃用警告和工件上传成功信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30524819759/job/90813290925) |

- **multimodal-gen-test-2-npu-a3**: 日志末尾显示工件成功上传，无测试失败或异常退出信息，可能因日志截断或作业实际成功但状态标记有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30524819759/job/90813290925

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524819759/job/90813290926) |


## [Run #30524785306](https://github.com/sgl-project/sglang/actions/runs/30524785306)
- **分支**: `cp/interleave-v2`
- **总耗时**: 43.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30524785306

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186246) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186252) |
| stage-b-test-1-npu-a2 (0) | 42.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186253) |
| stage-b-test-1-npu-a2 (1) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186257) |
| stage-b-test-4-npu-a3 | 40.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186260) |
| multimodal-gen-test-1-npu-a3 | 35.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186264) |
| stage-b-test-2-npu-a2 (1) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186269) |
| stage-b-test-16-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186281) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186305) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30524785306/job/90813186692) |


---
*Auto-generated by npu_pr_monitor.py*