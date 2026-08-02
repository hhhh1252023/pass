# NPU CI 执行监控
**生成时间**: 2026-08-02 12:28 UTC
**分析 Run 数**: 25

---

## [Run #30746310716](https://github.com/sgl-project/sglang/actions/runs/30746310716)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 16.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30746310716

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450440) |
| stage-b-test-16-npu-a3 | 15.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450443) |
| stage-b-test-2-npu-a2 (0) | 14.1min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450451) |
| multimodal-gen-test-2-npu-a3 | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450453) |
| stage-b-test-1-npu-a2 (1) | 14.3min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450460) |
| stage-b-test-1-npu-a2 (0) | 16.0min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450462) |
| multimodal-gen-test-1-npu-a3 | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450465) |
| stage-b-test-2-npu-a2 (1) | 10.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450466) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450830) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450865) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450872) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450879) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 15.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450894) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450898) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450440

- **stage-b-test-16-npu-a3**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450443

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行正常（进度96%），但突然报错“Executing the custom container implementation failed”，提示联系runner管理员，属于runner环境或容器执行问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450451

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450453

- **stage-b-test-1-npu-a2 (1)**: 作业在运行第二个测试时，自定义容器实现执行失败，导致测试进程被终止。日志显示"Executing the custom container implementation failed"，属于自托管runner环境问题，而非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450460

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到32%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU环境或容器问题导致作业中断，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450462

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450465

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试进行到97%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是runner环境或容器问题导致作业中断，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450466

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450830

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450865

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450872

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450879

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程存储对象缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源缺失问题，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450894

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450898

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30746310716/job/91492450439) |


## [Run #30746057594](https://github.com/sgl-project/sglang/actions/runs/30746057594)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 7.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30746057594

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787462) |
| multimodal-gen-test-1-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787463) |
| stage-b-test-16-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787464) |
| multimodal-gen-test-2-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787469) |
| stage-b-test-2-npu-a2 (1) | 5.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787470) |
| stage-b-test-4-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787471) |
| stage-b-test-2-npu-a2 (0) | 5.4min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787472) |
| stage-b-test-1-npu-a2 (0) | 7.1min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787480) |
| stage-b-test-1-npu-a2 (1) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787485) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787714) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787732) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 7.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787748) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787750) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787755) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787766) |

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787462

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787463

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787464

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787469

- **stage-b-test-2-npu-a2 (1)**: 作业在TokenizerManager初始化后，执行自定义容器实现时失败，报错提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787470

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787471

- **stage-b-test-2-npu-a2 (0)**: 作业在加载模型权重时（Multi-thread loading shards 0%）自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787472

- **stage-b-test-1-npu-a2 (0)**: 日志显示BlobNotFound错误，说明CI系统尝试下载的日志文件在Azure Blob存储中不存在，可能是日志上传失败、路径错误或文件被清理，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787480

- **stage-b-test-1-npu-a2 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储中找到，可能是资源被清理、路径错误或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787485

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）未上传或已被删除，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787714

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787732

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未上传或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787748

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787750

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储资源缺失或路径错误，可能是配置问题或资源被清理，需检查相关 blob 路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787755

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30746057594/job/91491787766


## [Run #30744611078](https://github.com/sgl-project/sglang/actions/runs/30744611078)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 42.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30744611078

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 42.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035914) |
| multimodal-gen-test-2-npu-a3 | 42.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035916) |
| multimodal-gen-test-1-npu-a3 | 42.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035943) |
| stage-b-test-4-npu-a3 | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035984) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 42.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036168) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 42.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036187) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036190) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036191) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036195) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 42.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036216) |

- **stage-b-test-16-npu-a3**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误（HTTP 404），说明该文件已被删除、路径错误或尚未上传，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035914

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业依赖的某个blob资源已被删除或路径错误，可能是数据未上传、过期或配置的URL有误，需检查相关存储路径及权限。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035916

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的某个 blob 不存在，可能是日志文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035943

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035984

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是模型权重、测试数据或中间产物未正确上传或路径配置错误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036168

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被删除、路径错误或上传未完成。需检查CI配置中的blob路径或重新上传相关文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036187

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是测试数据或日志未正确上传，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036190

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036191

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036195

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传或已被清理，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488036216

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035919) |
| stage-b-test-2-npu-a2 (1) | 11.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035921) |
| stage-b-test-1-npu-a2 (0) | 35.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035927) |
| stage-b-test-1-npu-a2 (1) | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035928) |
| stage-b-test-2-npu-a2 (0) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744611078/job/91488035930) |


## [Run #30744550998](https://github.com/sgl-project/sglang/actions/runs/30744550998)
- **分支**: `kan/rust-server-pd-restack`
- **总耗时**: 40.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30744550998

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876620) |
| stage-b-test-4-npu-a3 | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876621) |
| multimodal-gen-test-2-npu-a3 | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876635) |
| multimodal-gen-test-1-npu-a3 | 39.7min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876639) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876983) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876998) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487877002) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487877016) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487877035) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 39.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487877036) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876620

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876621

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876635

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876639

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理或配置变更，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876983

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败或配置变更，需检查相关 blob 是否存在及路径配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876998

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的 Azure Blob 存储资源缺失或路径错误，可能是数据未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487877002

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的模型或数据文件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487877016

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置或重新上传数据。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487877035

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被误删、路径错误或上传未完成。这属于外部依赖环境问题，需检查CI配置中的blob路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487877036

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876638) |
| stage-b-test-2-npu-a2 (1) | 11.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876641) |
| stage-b-test-1-npu-a2 (1) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876647) |
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876652) |
| stage-b-test-1-npu-a2 (0) | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744550998/job/91487876663) |


## [Run #30744255503](https://github.com/sgl-project/sglang/actions/runs/30744255503)
- **分支**: `kan/rust-server-pd-restack`
- **总耗时**: 9.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30744255503

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060638) |
| stage-b-test-2-npu-a2 (1) | 6.3min | 环境问题 | NPU测试服务启动后健康检查返回503，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060642) |
| multimodal-gen-test-1-npu-a3 | 8.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060648) |
| stage-b-test-1-npu-a2 (0) | 6.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060654) |
| multimodal-gen-test-2-npu-a3 | 8.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060655) |
| stage-b-test-1-npu-a2 (1) | 7.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060656) |
| stage-b-test-2-npu-a2 (0) | 6.6min | 环境问题 | NPU测试环境健康检查失败，服务返回503导致容器执行中止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060662) |
| stage-b-test-16-npu-a3 | 8.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060666) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 8.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060877) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060886) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060924) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 8.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060931) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060935) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060936) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060638

- **stage-b-test-2-npu-a2 (1)**: 日志显示服务在启动过程中多次健康检查返回503 Service Unavailable，最终容器执行失败。可能原因是NPU环境初始化慢或资源不足，导致服务未能及时就绪。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060642

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储资源缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060648

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试在运行约12秒后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060654

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的远程资源（如模型权重、测试数据或缓存）在存储账户中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060655

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060656

- **stage-b-test-2-npu-a2 (0)**: 日志显示服务启动后/health_generate接口返回503 Service Unavailable，随后自定义容器执行失败。可能是NPU资源初始化异常或服务未就绪，属于环境问题而非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060662

- **stage-b-test-16-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060666

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业在尝试访问Azure Blob存储中的某个blob时，返回BlobNotFound错误，说明该blob已被删除或路径错误，属于环境或资源缺失问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060877

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060886

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060924

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未上传或路径错误，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060931

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的模型/数据文件在 Azure Blob 存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060935

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060936

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744255503/job/91487060635) |


## [Run #30744184110](https://github.com/sgl-project/sglang/actions/runs/30744184110)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 11.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30744184110

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911796) |
| multimodal-gen-test-1-npu-a3 | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911798) |
| stage-b-test-16-npu-a3 | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911802) |
| stage-b-test-1-npu-a2 (1) | 9.1min | 环境问题 | NPU测试执行过程中自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911803) |
| stage-b-test-1-npu-a2 (0) | 9.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911804) |
| multimodal-gen-test-2-npu-a3 | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911807) |
| stage-b-test-2-npu-a2 (1) | 9.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911810) |
| stage-b-test-2-npu-a2 (0) | 9.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911817) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 10.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912074) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 10.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912077) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 10.2min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912106) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 10.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912114) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912129) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 10.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912130) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911796

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911798

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911802

- **stage-b-test-1-npu-a2 (1)**: 日志显示在运行test_npu_piecewise_graph_prefill.py时，出现'Executing the custom container implementation failed'错误，可能是NPU环境或容器配置问题，而非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911803

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到76%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911804

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911807

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行中突然出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911810

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行约9分钟后，在Prefill batch处理时出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911817

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 错误码BlobNotFound表明CI尝试下载或访问的远程资源（如模型权重或测试数据）在存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912074

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被删除、路径错误或上传未完成。这属于外部依赖（存储服务）问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912077

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示BlobNotFound错误，说明CI作业尝试访问的Azure Blob存储资源缺失或路径错误，可能是配置问题或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912106

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误信息为BlobNotFound，说明作业依赖的某个文件或资源在Azure Blob存储中缺失，可能是上传失败、路径错误或资源被删除，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912114

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）未找到，可能是资源被删除、路径错误或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912129

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件（如模型权重或数据）在 Azure Blob 存储中缺失或路径错误，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486912130

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30744184110/job/91486911794) |


## [Run #30743900542](https://github.com/sgl-project/sglang/actions/runs/30743900542)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 9.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30743900542

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129970) |
| multimodal-gen-test-2-npu-a3 | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129972) |
| stage-b-test-4-npu-a3 | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129980) |
| stage-b-test-2-npu-a2 (1) | 7.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129991) |
| stage-b-test-16-npu-a3 | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129997) |
| stage-b-test-2-npu-a2 (0) | 7.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129998) |
| stage-b-test-1-npu-a2 (1) | 7.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130022) |
| stage-b-test-1-npu-a2 (0) | 7.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130023) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130225) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130234) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130257) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 8.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130270) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130275) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130293) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129970

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中缺失，可能是资源被清理或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129972

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129980

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行中突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129991

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 中已被删除或路径错误，属于环境配置或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129997

- **stage-b-test-2-npu-a2 (0)**: 日志显示Prefill批处理正常进行，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129998

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130022

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行中突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130023

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130225

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure 存储资源缺失或路径错误，可能是数据未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130234

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或测试数据未上传到指定存储路径，或路径配置错误，需检查相关资源是否就绪。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130257

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的 blob 已被删除或路径错误，可能是日志清理或配置问题，需检查存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130270

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130275

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486130293

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743900542/job/91486129976) |


## [Run #30743697886](https://github.com/sgl-project/sglang/actions/runs/30743697886)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 6.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30743697886

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 5.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572803) |
| stage-b-test-16-npu-a3 | 5.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572805) |
| stage-b-test-2-npu-a2 (0) | 5.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572812) |
| stage-b-test-2-npu-a2 (1) | 4.9min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572816) |
| multimodal-gen-test-2-npu-a3 | 5.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572830) |
| stage-b-test-1-npu-a2 (1) | 5.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572839) |
| stage-b-test-1-npu-a2 (0) | 5.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572856) |
| multimodal-gen-test-1-npu-a3 | 5.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572864) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 5.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573075) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573079) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 5.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573083) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 5.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573084) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 5.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573089) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 5.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573096) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572803

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572805

- **stage-b-test-2-npu-a2 (0)**: 作业在启动自定义容器时失败，日志显示torch_npu导入时出现警告，随后报错“Executing the custom container implementation failed”，属于NPU运行环境配置或容器启动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572812

- **stage-b-test-2-npu-a2 (1)**: 作业在启动NPU推理服务时，TokenizerManager初始化后自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572816

- **multimodal-gen-test-2-npu-a3**: 作业在下载或访问 Azure Blob 中的日志文件时，返回 BlobNotFound 错误，说明该文件已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572830

- **stage-b-test-1-npu-a2 (1)**: 作业在启动NPU测试容器时失败，日志显示torch_npu初始化后出现"Executing the custom container implementation failed"错误，可能是容器或NPU驱动环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572839

- **stage-b-test-1-npu-a2 (0)**: 作业在启动TokenizerManager后报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572856

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源缺失或路径错误，可能是上传失败、文件被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572864

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明作业依赖的某个数据文件或资源在 Azure Blob 存储中缺失，可能是上传失败、路径错误或文件被删除，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573075

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）未上传或已被删除，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573079

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573083

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573084

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573089

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业失败原因是Azure Blob存储返回BlobNotFound错误，即请求的资源不存在。这可能是由于文件被删除、路径错误或存储配置问题，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485573096

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743697886/job/91485572817) |


## [Run #30743537095](https://github.com/sgl-project/sglang/actions/runs/30743537095)
- **分支**: `mmangkad/fix-dsv4-dspark-hybrid-nvfp4`
- **总耗时**: 25.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30743537095

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116310) |
| stage-b-test-4-npu-a3 | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116320) |
| stage-b-test-1-npu-a2 (0) | 23.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116341) |
| stage-b-test-1-npu-a2 (1) | 23.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116343) |
| stage-b-test-16-npu-a3 | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116344) |
| multimodal-gen-test-2-npu-a3 | 24.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116350) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 24.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116597) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116632) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116648) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116659) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116664) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116680) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116310

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116320

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常（HTTP 200，进度83%），但突然报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116341

- **stage-b-test-1-npu-a2 (1)**: 日志显示服务启动成功并响应请求，但随后出现'Executing the custom container implementation failed'错误，属于自托管runner环境问题，非代码或测试逻辑失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116343

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的 blob 路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116344

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的依赖文件或工件在存储中缺失，可能是上传失败、路径错误或文件被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116350

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是数据未上传、路径错误或存储被清理。这属于外部依赖环境问题，需检查数据准备或存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116597

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116632

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116648

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116659

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116664

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116680

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116314) |
| stage-b-test-2-npu-a2 (1) | 11.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116345) |
| stage-b-test-2-npu-a2 (0) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743537095/job/91485116349) |


## [Run #30743117349](https://github.com/sgl-project/sglang/actions/runs/30743117349)
- **分支**: `kan/rust-server-hf-cache-fix`
- **总耗时**: 13.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30743117349

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 12.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970296) |
| stage-b-test-16-npu-a3 | 12.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970301) |
| stage-b-test-1-npu-a2 (1) | 12.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970305) |
| multimodal-gen-test-1-npu-a3 | 12.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970315) |
| stage-b-test-1-npu-a2 (0) | 11.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970316) |
| stage-b-test-2-npu-a2 (0) | 12.3min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970318) |
| multimodal-gen-test-2-npu-a3 | 12.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970331) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 12.5min | 环境问题 | Azure Blob 存储中找不到指定文件，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970496) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 12.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970512) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 12.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970515) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 12.5min | 环境问题 | 日志显示Azure Blob存储中找不到指定文件，属于外部依赖缺失。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970521) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 12.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970528) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 12.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970547) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970296

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期清理或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970301

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于基础设施/容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970305

- **multimodal-gen-test-1-npu-a3**: 作业在尝试下载或访问某个blob时，返回BlobNotFound错误，可能是CI配置中引用的文件路径错误或文件已被删除，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970315

- **stage-b-test-1-npu-a2 (0)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，错误提示联系自托管runner管理员，属于NPU环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970316

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行到59%时，自定义容器执行失败（Executing the custom container implementation failed），可能是NPU资源或容器环境问题导致作业中断，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970318

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970331

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在存储中不存在，可能是文件被误删或路径配置错误，需检查相关资源是否就绪。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970496

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970512

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970515

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业在拉取或访问测试所需的数据/模型文件时，Azure Blob返回BlobNotFound错误，说明文件不存在或路径错误，需检查存储配置或文件是否被清理。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970521

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是数据未上传或已被删除，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970528

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970547

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970306) |
| stage-b-test-2-npu-a2 (1) | 11.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743117349/job/91483970321) |


## [Run #30743100681](https://github.com/sgl-project/sglang/actions/runs/30743100681)
- **分支**: `codex/native-video-audio-pipeline`
- **总耗时**: 15.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30743100681

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988375) |
| stage-b-test-16-npu-a3 | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988376) |
| stage-b-test-1-npu-a2 (0) | 14.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988378) |
| stage-b-test-1-npu-a2 (1) | 14.2min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988392) |
| multimodal-gen-test-2-npu-a3 | 14.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988393) |
| multimodal-gen-test-1-npu-a3 | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988395) |
| stage-b-test-2-npu-a2 (0) | 14.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988405) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988508) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988540) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 14.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988542) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 14.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988574) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988583) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 14.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988592) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988375

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988376

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988378

- **stage-b-test-1-npu-a2 (1)**: 作业在加载模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），导致测试提前终止。可能是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988392

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988393

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988395

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试进行到约98%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988405

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源是否存在或更新引用路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988508

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988540

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 错误码BlobNotFound表明CI作业尝试访问的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988542

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业日志返回BlobNotFound错误，表明CI系统尝试访问的Azure Blob存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988574

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988583

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明作业依赖的 Azure Blob 存储资源缺失或路径错误，可能是数据未上传、被删除或配置错误，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988592

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988377) |
| stage-b-test-2-npu-a2 (1) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30743100681/job/91483988401) |


## [Run #30741518783](https://github.com/sgl-project/sglang/actions/runs/30741518783)
- **分支**: `cheng/gc-wb-stack-review`
- **总耗时**: 17.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30741518783

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 16.6min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753716) |
| stage-b-test-16-npu-a3 | 16.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753726) |
| multimodal-gen-test-1-npu-a3 | 16.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753730) |
| stage-b-test-4-npu-a3 | 16.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753733) |
| stage-b-test-2-npu-a2 (0) | 16.6min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753746) |
| stage-b-test-1-npu-a2 (0) | 15.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753748) |
| multimodal-gen-test-2-npu-a3 | 16.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753753) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 16.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754127) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754129) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 16.8min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754137) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 16.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754139) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 16.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754146) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 16.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754153) |

- **stage-b-test-1-npu-a2 (1)**: 日志显示模型权重加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753716

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于环境或资源配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753726

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753730

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753733

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试进行到99%时，runner报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753746

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但在执行过程中出现错误：'Executing the custom container implementation failed. Please contact your self hosted runner administrator.'，这属于自托管运行器环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753748

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业依赖的某个blob（可能是模型权重或数据文件）在存储中缺失，可能是文件被删除、路径错误或上传失败，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753753

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程存储对象缺失，可能是文件被删除、路径错误或上传失败，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754127

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754129

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI作业依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754137

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754139

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754146

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479754153

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753723) |
| stage-b-test-2-npu-a2 (1) | 10.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30741518783/job/91479753763) |


## [Run #30741048548](https://github.com/sgl-project/sglang/actions/runs/30741048548)
- **分支**: `cheng/gc-wb-stack-review`
- **总耗时**: 12.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30741048548

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605581) |
| multimodal-gen-test-1-npu-a3 | 12.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605606) |
| stage-b-test-1-npu-a2 (0) | 11.0min | 环境问题 | 自定义容器执行失败，NPU环境初始化或资源分配异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605607) |
| stage-b-test-1-npu-a2 (1) | 9.4min | 环境问题 | NPU测试执行过程中自定义容器实现失败，作业被强制终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605610) |
| stage-b-test-16-npu-a3 | 12.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605616) |
| stage-b-test-2-npu-a2 (1) | 9.1min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605619) |
| stage-b-test-2-npu-a2 (0) | 11.8min | 环境问题 | 自托管runner执行自定义容器实现失败，作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605633) |
| multimodal-gen-test-2-npu-a3 | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605651) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605957) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605967) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 12.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605986) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 12.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605996) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478606017) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 12.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478606030) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605581

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的依赖文件或工件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605606

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重时（Multi-thread loading shards 0%）自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605607

- **stage-b-test-1-npu-a2 (1)**: 在运行test_npu_piecewise_graph_prefill.py时，执行到prefill latency测试后，自定义容器实现报错（Executing the custom container implementation failed），导致作业中断，非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605610

- **stage-b-test-16-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的blob已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605616

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行到47%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605619

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行中（进度54%）时，runner报错“Executing the custom container implementation failed”，属于自托管runner环境或容器问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605633

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或存储配置变更，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605651

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源是否存在或更新引用路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605957

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605967

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的远程存储对象缺失，可能是文件被误删、路径配置错误或上传失败，需检查CI配置中的存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605986

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是文件被删除、路径错误或上传未完成。这属于外部依赖缺失，非代码或性能问题，需检查CI配置中的blob路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605996

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478606017

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明CI依赖的远程存储文件缺失或路径错误，可能是上传失败、文件被删除或配置的URL有误，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478606030

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30741048548/job/91478605590) |


## [Run #30740788317](https://github.com/sgl-project/sglang/actions/runs/30740788317)
- **分支**: `feat/srt-empty`
- **总耗时**: 5.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30740788317

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801206) |
| multimodal-gen-test-2-npu-a3 | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801209) |
| multimodal-gen-test-1-npu-a3 | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801214) |
| stage-b-test-1-npu-a2 (0) | 3.5min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801216) |
| stage-a-unit-test-npu | 2.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801217) |
| stage-b-test-2-npu-a2 (1) | 3.2min | 环境问题 | 自定义容器执行失败，构建xatlas依赖时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801219) |
| stage-b-test-16-npu-a3 | 3.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801221) |
| stage-b-test-1-npu-a2 (1) | 3.1min | 环境问题 | 自定义容器执行失败，构建xatlas依赖时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801224) |
| stage-b-test-2-npu-a2 (0) | 2.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801226) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801506) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 3.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801525) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801564) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801568) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 3.8min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801592) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801596) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801206

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801209

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801214

- **stage-b-test-1-npu-a2 (0)**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801216

- **stage-a-unit-test-npu**: 日志显示在安装自定义算子包后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于NPU环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801217

- **stage-b-test-2-npu-a2 (1)**: 作业在构建xatlas Python包时，自定义容器实现执行失败（Executing the custom container implementation failed），导致构建流程中断，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801219

- **stage-b-test-16-npu-a3**: 作业在下载或访问某个blob时失败，返回BlobNotFound错误。可能是文件被删除、路径错误或上传未完成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801221

- **stage-b-test-1-npu-a2 (1)**: 作业在构建xatlas 0.0.11依赖时，自定义容器实现执行失败，导致构建中断。日志显示编译进行到第4步时出错，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801224

- **stage-b-test-2-npu-a2 (0)**: 日志显示在安装triton-ascend依赖时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801226

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查资源是否存在或更新引用路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801506

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明作业依赖的某个blob（可能是模型权重、测试数据或日志文件）已被删除或路径错误，需检查CI配置中的存储路径或资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801525

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801564

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，可能是测试所需的模型权重或数据文件未正确上传到存储，或路径配置错误，需检查 CI 作业的依赖资源是否完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801568

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI系统尝试下载的日志或工件在Azure Blob存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801592

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740788317/job/91477801596


## [Run #30740213418](https://github.com/sgl-project/sglang/actions/runs/30740213418)
- **分支**: `main`
- **总耗时**: 199.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30740213418

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 138.9min | 其他 | 日志被截断，无法看到实际失败原因，仅显示作业结束时的清理和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184535) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 14.9min | 其他 | 日志被截断，未显示测试执行结果，无法判断具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184637) |
| single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms | 22.4min | 其他 | 日志被截断，无法确定具体失败原因 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91483001574) |
| single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms | 19.7min | 其他 | 日志被截断，未显示测试执行结果，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91483528853) |
| single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms | 9.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91493000046) |
| single-node-poc (qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu... / qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms | 7.3min | 环境问题 | 依赖的Blob存储文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91493169187) |
| single-node-poc (glm4_7_flash_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/glm4_... / glm4_7_flash_1p_gsm8k | 3.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91493512266) |

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志中间部分被省略，未包含测试执行或失败的具体错误信息。仅看到作业结束时的plog备份和artifact上传步骤，以及Node.js 20弃用警告，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184535

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志仅包含作业初始化和清理阶段，中间部分被省略，未出现测试命令、错误信息或性能数据，无法定位失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184637

- **single-node-poc (minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-8, test/registe... / minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms**: 日志仅包含作业启动和清理信息，中间关键测试执行部分被省略，未显示错误或失败点。需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91483001574

- **single-node-poc (kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms, linux-aarch64-a3-16, test/registered/npu/pe... / kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms**: 日志仅包含作业启动、环境准备和清理信息，未展示测试运行过程及错误输出，可能因日志截断或作业在测试前被中断，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91483528853

- **single-node-poc (qwen3_235b_w8a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu/p... / qwen3_235b_w8a8_8p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91493000046

- **single-node-poc (qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms, linux-aarch64-a3-16, test/registered/npu... / qwen3_5_397b_w4a8_8p_in3k5_out1k5_50ms**: 日志显示Azure Blob存储返回BlobNotFound错误，说明作业所需的模型权重或数据文件未上传或已被删除，属于环境配置或资源缺失问题，需检查CI依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91493169187

- **single-node-poc (glm4_7_flash_1p_gsm8k, linux-aarch64-a3-2, test/registered/ascend/accuracy/glm4_... / glm4_7_flash_1p_gsm8k**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91493512266

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184281) |
| stage-b-test-1-npu-a2 (1) | 32.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184283) |
| stage-b-test-1-npu-a2 (0) | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184284) |
| stage-b-test-2-npu-a2 (1) | 11.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184292) |
| stage-b-test-2-npu-a2 (0) | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184295) |
| stage-b-test-16-npu-a3 | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184297) |
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184301) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184543) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184589) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184615) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91476184619) |
| single-node-poc (qwen3_32b_w8a8_2p_in3k5_out1k5_50ms, linux-aarch64-a3-4, test/registered/npu/per... / qwen3_32b_w8a8_2p_in3k5_out1k5_50ms | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91482839121) |
| single-node-poc (qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16, linux-aarch64-a3-4, test/registered/npu... / qwen3_next_80b_w8a8_2p_in6k_out1k5_bs16 | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91482936589) |
| single-node-poc (deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms, linux-aarch64-a3-16, test/registered/... / deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740213418/job/91483247081) |


## [Run #30740007668](https://github.com/sgl-project/sglang/actions/runs/30740007668)
- **分支**: `cheng/gc-global-read-sweep`
- **总耗时**: 23.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30740007668

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (0) | 20.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651716) |
| stage-b-test-16-npu-a3 | 22.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651717) |
| multimodal-gen-test-1-npu-a3 | 22.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651720) |
| stage-b-test-2-npu-a2 (0) | 20.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651730) |
| multimodal-gen-test-2-npu-a3 | 22.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651736) |
| stage-b-test-1-npu-a2 (1) | 19.8min | 超时 | Scheduler watchdog 超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651748) |
| stage-b-test-4-npu-a3 | 22.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651752) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 22.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651961) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 22.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651987) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 22.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651988) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 22.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475652012) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 22.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475652035) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475652047) |

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651716

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是上游产物未上传或存储配置变更，需检查相关依赖文件是否生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651717

- **multimodal-gen-test-1-npu-a3**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651720

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行中（Prefill batch正常处理），但突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651730

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651736

- **stage-b-test-1-npu-a2 (1)**: 日志显示 scheduler watchdog 超时（300秒），pyspy 转储线程卡在 subprocess 通信，最终自定义容器执行失败，属于测试超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651748

- **stage-b-test-4-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的存储对象已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651752

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 错误码BlobNotFound表明CI依赖的远程存储资源缺失或路径错误，可能是模型权重、测试数据或中间产物未正确上传，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651961

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的远程资源（如模型权重或数据文件）在存储中缺失或路径错误，属于环境配置或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651987

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程数据文件缺失或路径错误，可能是数据未上传、被删除或配置的 URL 有误，需检查数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651988

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储账户中缺失，可能是文件被删除、路径错误或上传未完成。建议检查CI配置中的blob路径及上传步骤，确认文件存在且权限正确。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475652012

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475652035

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475652047

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651715) |
| stage-b-test-2-npu-a2 (1) | 11.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007668/job/91475651742) |


## [Run #30740007308](https://github.com/sgl-project/sglang/actions/runs/30740007308)
- **分支**: `cheng/gc-wb-3-control-plane`
- **总耗时**: 23.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30740007308

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628730) |
| stage-b-test-16-npu-a3 | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628731) |
| stage-b-test-1-npu-a2 (1) | 22.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628732) |
| multimodal-gen-test-1-npu-a3 | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628735) |
| stage-b-test-1-npu-a2 (0) | 22.4min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628738) |
| multimodal-gen-test-2-npu-a3 | 23.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628755) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628902) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628932) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 23.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628934) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628948) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628968) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 23.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628991) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628730

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628731

- **stage-b-test-1-npu-a2 (1)**: 日志显示sglang服务正常启动并处理请求，但随后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628732

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628735

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但在执行过程中出现“Executing the custom container implementation failed”错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628738

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的依赖文件或工件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628755

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或数据文件未上传或已被删除，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628902

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628932

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明作业依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题，需检查CI脚本中的下载路径或资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628934

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628948

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628968

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明CI作业尝试下载的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源管理问题，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628991

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628721) |
| stage-b-test-2-npu-a2 (1) | 11.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628724) |
| stage-b-test-2-npu-a2 (0) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007308/job/91475628727) |


## [Run #30740007141](https://github.com/sgl-project/sglang/actions/runs/30740007141)
- **分支**: `cheng/gc-wb-4-fpm-endpoint`
- **总耗时**: 23.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30740007141

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 22.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644634) |
| multimodal-gen-test-1-npu-a3 | 22.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644639) |
| stage-b-test-1-npu-a2 (1) | 20.6min | 超时 | Scheduler watchdog 超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644649) |
| stage-b-test-2-npu-a2 (0) | 20.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644652) |
| multimodal-gen-test-2-npu-a3 | 22.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644657) |
| stage-b-test-1-npu-a2 (0) | 20.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化或资源分配异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644661) |
| stage-b-test-4-npu-a3 | 22.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644667) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 22.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644813) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 22.9min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644822) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 22.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644828) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 22.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644842) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 22.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644851) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 22.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644857) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644634

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644639

- **stage-b-test-1-npu-a2 (1)**: 日志显示 Scheduler watchdog 在 300 秒内未收到调度器响应，触发软超时，随后自定义容器执行失败，作业终止。可能原因是 NPU 资源竞争或调度器卡死。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644649

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试进行到92%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644652

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644657

- **stage-b-test-1-npu-a2 (0)**: 作业在加载模型权重时（Multi-thread loading shards 0%）突然报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644661

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 资源已被删除或路径错误，可能是上游产物未上传或存储配置变更，需检查相关依赖文件是否正常生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644667

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644813

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示BlobNotFound错误，可能是CI脚本尝试下载或访问的模型/数据文件在存储中缺失或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644822

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的远程数据文件缺失或路径错误，可能是资源被清理或配置变更，需检查存储路径或重新上传数据。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644828

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试从 Azure Blob 下载日志文件时，该文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644842

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644851

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件或数据在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644857

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644644) |
| stage-b-test-2-npu-a2 (1) | 11.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740007141/job/91475644650) |


## [Run #30740005990](https://github.com/sgl-project/sglang/actions/runs/30740005990)
- **分支**: `cheng/gc-wb-2-draft-copy`
- **总耗时**: 23.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30740005990

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629178) |
| stage-b-test-1-npu-a2 (0) | 22.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629199) |
| stage-b-test-1-npu-a2 (1) | 22.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629200) |
| multimodal-gen-test-2-npu-a3 | 23.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629202) |
| stage-b-test-16-npu-a3 | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629207) |
| multimodal-gen-test-1-npu-a3 | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629208) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629448) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629459) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629463) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 23.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629474) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 23.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629483) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 23.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629500) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629178

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到42%时，自定义容器实现执行失败（Executing the custom container implementation failed），导致作业中断，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629199

- **stage-b-test-1-npu-a2 (1)**: 日志显示在捕获批次数据后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629200

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储对象已被删除或路径错误，可能是CI配置中引用的模型权重或缓存文件缺失，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629202

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629207

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629208

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的模型权重或测试数据未上传到指定存储路径，或路径配置错误，需检查相关资源是否就绪。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629448

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629459

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629463

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储中缺失，可能是CI流程中引用的工件或日志文件被清理、路径错误或上传失败，需检查相关存储配置和文件生命周期。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629474

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629483

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629500

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629189) |
| stage-b-test-2-npu-a2 (0) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629204) |
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30740005990/job/91475629206) |


## [Run #30739593153](https://github.com/sgl-project/sglang/actions/runs/30739593153)
- **分支**: `feat/srt-empty`
- **总耗时**: 37.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30739593153

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 35.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547632) |
| multimodal-gen-test-2-npu-a3 | 1.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547648) |
| multimodal-gen-test-1-npu-a3 | 7.0min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547650) |
| stage-b-test-16-npu-a3 | 35.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547652) |
| stage-b-test-1-npu-a2 (0) | 34.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547666) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 35.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547948) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 35.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547953) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 35.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547967) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 35.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547996) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 35.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547999) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 35.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474548014) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547632

- **multimodal-gen-test-2-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547648

- **multimodal-gen-test-1-npu-a3**: 日志截断，缺少测试执行部分，无法判断失败原因。仅见Node 20弃用警告和上传artifact时无文件，可能测试未运行或提前退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547650

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547652

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547666

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的模型或数据文件在 Azure Blob 中缺失，可能是文件被误删或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547948

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，可能是测试所需的数据或模型文件未上传到指定存储位置，或路径配置错误，需检查 CI 数据准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547953

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547967

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业日志返回BlobNotFound错误，表明CI系统尝试访问的Azure Blob存储资源（可能为模型权重或测试数据）不存在或已被删除，属于环境配置或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547996

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547999

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 错误码BlobNotFound表明CI作业尝试下载的模型或数据文件在存储中缺失，可能是文件被误删、路径配置错误或上传未完成，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474548014

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547638) |
| stage-b-test-2-npu-a2 (1) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547657) |
| stage-b-test-2-npu-a2 (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547662) |
| stage-b-test-1-npu-a2 (1) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30739593153/job/91474547669) |


## [Run #30739329824](https://github.com/sgl-project/sglang/actions/runs/30739329824)
- **分支**: `feat/srt-empty`
- **总耗时**: 8.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30739329824

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841450) |
| stage-b-test-4-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841453) |
| stage-b-test-2-npu-a2 (0) | 5.5min | 环境问题 | 自定义容器执行失败，模型加载权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841454) |
| stage-b-test-2-npu-a2 (1) | 6.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841458) |
| stage-b-test-16-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841461) |
| stage-b-test-1-npu-a2 (1) | 5.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841471) |
| multimodal-gen-test-2-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841477) |
| stage-b-test-1-npu-a2 (0) | 6.9min | 环境问题 | NPU后端算子不支持导致服务启动后健康检查失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841478) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 7.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841597) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841625) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 7.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841639) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841657) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841666) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 7.1min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841683) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或文件被清理，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841450

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，属于环境/资源缺失问题，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841453

- **stage-b-test-2-npu-a2 (0)**: 作业在加载模型权重（Multi-thread loading shards）时，自定义容器实现执行失败，提示联系自托管runner管理员，可能因NPU环境或容器配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841454

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行中（进度3/1319）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841458

- **stage-b-test-16-npu-a3**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是文件被误删、路径错误或上传失败，属于外部存储环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841461

- **stage-b-test-1-npu-a2 (1)**: 作业在torch分布式初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841471

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841477

- **stage-b-test-1-npu-a2 (0)**: 日志显示NPU后端不支持aten::_assert_async算子，回退到CPU执行，导致服务启动后health_generate接口持续返回503，最终自定义容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841478

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841597

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在指定容器中不存在，可能是文件被删除、路径错误或上传未完成，需检查 CI 配置中的 blob 路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841625

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储账户中缺失，可能是文件被删除、路径错误或上传未完成。建议检查CI配置中的blob路径或重新上传相关文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841639

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841657

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841666

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI作业依赖的远程日志文件已被删除或路径错误，属于环境配置或资源缺失问题，需检查存储路径或重新上传日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841683

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30739329824/job/91473841459) |


## [Run #30739240264](https://github.com/sgl-project/sglang/actions/runs/30739240264)
- **分支**: `main`
- **总耗时**: 10.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30739240264

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取必要数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576924) |
| stage-b-test-1-npu-a2 (1) | 7.1min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576934) |
| stage-b-test-2-npu-a2 (0) | 7.3min | 环境问题 | 自定义容器执行失败，NPU后端不支持CUDA相关操作导致服务启动异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576937) |
| stage-b-test-2-npu-a2 (1) | 7.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576948) |
| stage-b-test-1-npu-a2 (0) | 7.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576950) |
| stage-b-test-4-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576956) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577317) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 9.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577336) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 9.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577360) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577362) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577369) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577383) |

- **stage-b-test-16-npu-a3**: 作业 stage-b-test-16-npu-a3 在尝试下载或访问 Azure Blob 中的某个 blob 时，返回 BlobNotFound 错误（HTTP 404）。这通常是因为日志文件被清理、路径错误或上传未完成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576924

- **stage-b-test-1-npu-a2 (1)**: 日志显示测试正常运行至08:20:14，随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于NPU容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576934

- **stage-b-test-2-npu-a2 (0)**: 日志显示SymmetricMemory不支持cuda设备类型，且NPU后端存在算子回退警告，最终自定义容器实现执行失败，属于环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576937

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试正常运行中，但突然出现'Executing the custom container implementation failed'错误，提示联系runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576948

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576950

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576956

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577317

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 错误码BlobNotFound表明CI作业尝试下载的远程资源（如模型权重或测试数据）在存储账户中缺失，可能是文件被误删、路径配置错误或上传未完成，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577336

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577360

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是配置问题或文件被清理，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577362

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程数据文件缺失或路径错误，可能是存储配置变更或文件被清理，需检查相关 blob 是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577369

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473577383

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-a-unit-test-npu | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30739240264/job/91473576931) |


## [Run #30739094511](https://github.com/sgl-project/sglang/actions/runs/30739094511)
- **分支**: `feat/srt-empty`
- **总耗时**: 7.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30739094511

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-2-npu-a2 (0) | 2.7min | 环境问题 | 下载依赖包时网络连接失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189323) |
| stage-b-test-16-npu-a3 | 6.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189325) |
| multimodal-gen-test-2-npu-a3 | 6.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189331) |
| multimodal-gen-test-1-npu-a3 | 6.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189335) |
| stage-b-test-2-npu-a2 (1) | 2.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189336) |
| stage-b-test-4-npu-a3 | 6.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189343) |
| stage-b-test-1-npu-a2 (0) | 2.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189344) |
| stage-b-test-1-npu-a2 (1) | 2.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189346) |
| stage-a-unit-test-npu | 6.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189353) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 6.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189543) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 6.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189557) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 6.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189558) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 6.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189563) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 6.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189568) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 6.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189592) |

- **stage-b-test-2-npu-a2 (0)**: 在下载ops-transformer zip包时，连接gh-proxy.test.osinfra.cn超时，触发自定义容器执行失败，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189323

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189325

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储中缺失，可能是上传失败、路径错误或文件被清理，需检查相关存储配置和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189331

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或文件被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189335

- **stage-b-test-2-npu-a2 (1)**: 日志显示在安装依赖（triton等）后，出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189336

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189343

- **stage-b-test-1-npu-a2 (0)**: 日志显示在安装triton-ascend依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境配置或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189344

- **stage-b-test-1-npu-a2 (1)**: 日志显示在安装依赖后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189346

- **stage-a-unit-test-npu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189353

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件（如模型权重或测试数据）在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189543

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业日志中返回BlobNotFound错误，表明CI系统尝试访问的存储对象缺失或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189557

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189558

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189563

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到，可能是文件被删除、路径错误或上传失败，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189568

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30739094511/job/91473189592


## [Run #30737685878](https://github.com/sgl-project/sglang/actions/runs/30737685878)
- **分支**: `feat/srt-empty`
- **总耗时**: 43.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30737685878

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-a-unit-test-npu | 4.2min | 代码错误 | 测试文件缺少主入口导致测试收集失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392197) |
| multimodal-gen-test-1-npu-a3 | 42.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392202) |
| multimodal-gen-test-2-npu-a3 | 42.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392204) |
| stage-b-test-2-npu-a2 (1) | 4.0min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392213) |
| stage-b-test-1-npu-a2 (0) | 4.0min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392218) |
| stage-b-test-16-npu-a3 | 42.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392219) |
| stage-b-test-4-npu-a3 | 42.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392221) |
| stage-b-test-2-npu-a2 (0) | 3.4min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392231) |
| stage-b-test-1-npu-a2 (1) | 3.9min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392260) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 42.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392426) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 42.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392433) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 42.8min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392446) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 42.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需数据或模型文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392468) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 42.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392473) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 42.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392474) |

- **stage-a-unit-test-npu**: test_srt_empty_deps.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 脚本检测到该问题并抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392197

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问某个Azure Blob存储资源时，返回BlobNotFound错误，说明该资源已被删除或路径错误，属于环境或依赖资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392202

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392204

- **stage-b-test-2-npu-a2 (1)**: test_srt_empty_deps.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，run_suite.py 收集测试时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392213

- **stage-b-test-1-npu-a2 (0)**: test_srt_empty_deps.py未添加`if __name__ == "__main__":`入口，pytest风格测试在`python3 file.py -f`下会静默跳过，run_suite.py收集测试时抛出ValueError，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392218

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392219

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392221

- **stage-b-test-2-npu-a2 (0)**: 日志显示在创建PEP 517构建环境时，执行自定义容器实现失败（Executing the custom container implementation failed），提示联系runner管理员，属于runner环境或配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392231

- **stage-b-test-1-npu-a2 (1)**: test_srt_empty_deps.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 收集测试时抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392260

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 错误码BlobNotFound表明请求的资源在存储账户中缺失，可能是文件被误删、路径配置错误或上传未完成。建议检查CI配置中的blob路径及文件是否存在，并确认上传步骤是否成功。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392426

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误信息为BlobNotFound，说明作业依赖的某个文件或资源在存储中缺失，可能是配置错误或文件被删除，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392433

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 日志显示BlobNotFound错误，说明CI作业依赖的远程日志文件已被删除或路径错误，属于环境配置或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392446

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 错误码BlobNotFound表明CI依赖的远程存储资源缺失或路径错误，可能是数据未上传、被清理或配置有误，属于环境或资源准备问题，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392468

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或数据文件）未找到，可能是路径错误或资源被删除，需检查配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392473

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的模型/数据文件在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737685878/job/91469392474


## [Run #30737270375](https://github.com/sgl-project/sglang/actions/runs/30737270375)
- **分支**: `main`
- **总耗时**: 60.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/30737270375

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 59.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329232) |
| stage-b-test-4-npu-a3 | 59.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329238) |
| multimodal-gen-test-1-npu-a3 | 59.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329254) |
| multimodal-gen-test-2-npu-a3 | 59.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329260) |
| single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu | 59.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329449) |
| single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms | 59.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329458) |
| single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms | 59.6min | 环境问题 | 日志显示Azure Blob存储中找不到指定文件，属于外部依赖资源缺失。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329464) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 59.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329477) |
| single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms | 59.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329485) |
| single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms | 59.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329487) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329232

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329238

- **multimodal-gen-test-1-npu-a3**: 作业尝试下载日志文件时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329254

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或数据在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329260

- **single-node-poc (qwen3_vl_8b_thinking_1p_mmmu, linux-aarch64-a3-2, test/registered/npu/accuracy/q... / qwen3_vl_8b_thinking_1p_mmmu**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329449

- **single-node-poc (qwen3_30b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/per... / qwen3_30b_w8a8_1p_in3k5_out1k5_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329458

- **single-node-poc (qwen3_8b_w8a8_1p_in3k5_out1k5_50ms, linux-aarch64-a3-2, test/registered/npu/perf... / qwen3_8b_w8a8_1p_in3k5_out1k5_50ms**: 作业失败原因是下载或访问Azure Blob存储中的文件时返回BlobNotFound错误，即所需文件不存在或路径错误，与代码或性能无关，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329464

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/npu/pe... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误信息为BlobNotFound，表明作业尝试访问的Azure Blob存储资源缺失或路径错误，可能是CI配置中引用的模型权重或日志文件未正确上传，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329477

- **single-node-poc (qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms, linux-aarch64-a3-2, test/registere... / qwen3_6_35b_a3b_1p_in64k_out1k_prefix90_50ms**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/数据文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329485

- **single-node-poc (qwen3_6_27b_1p_in1024x1024_30_out1024_50ms, linux-aarch64-a3-2, test/registered/... / qwen3_6_27b_1p_in1024x1024_30_out1024_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329487

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329225) |
| stage-b-test-1-npu-a2 (1) | 33.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329229) |
| stage-b-test-2-npu-a2 (1) | 12.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329235) |
| stage-b-test-1-npu-a2 (0) | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329243) |
| stage-a-unit-test-npu | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/30737270375/job/91468329250) |


---
*Auto-generated by npu_pr_monitor.py*