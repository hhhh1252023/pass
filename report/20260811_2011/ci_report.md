# NPU CI 执行监控
**生成时间**: 2026-08-11 12:11 UTC
**分析 Run 数**: 16

---

## [Run #31484207064](https://github.com/sgl-project/sglang/actions/runs/31484207064)
- **分支**: `agent/optimize-sm120-ernie-norm-fusion`
- **总耗时**: 39.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31484207064

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 3.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916831) |
| base-b-test-4-npu-a3 / run (0) | 3.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916872) |
| base-b-test-8-npu-a3 / run (0) | 3.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916887) |
| base-b-test-2-npu-a3 / run (0) | 2.3min | 环境问题 | 下载依赖包时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916907) |
| base-b-test-16-npu-a3 / run (0) | 3.5min | 环境问题 | 自定义容器执行失败，NPU测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916915) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.5min | 环境问题 | 自定义容器执行失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917352) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 38.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917389) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 38.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917442) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917444) |

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，而非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916831

- **base-b-test-4-npu-a3 / run (0)**: 作业在运行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916872

- **base-b-test-8-npu-a3 / run (0)**: 作业在运行测试前，执行自定义容器实现时失败，错误提示联系runner管理员，属于基础设施/环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916887

- **base-b-test-2-npu-a3 / run (0)**: 作业在下载ops-transformer依赖包时，自定义容器实现执行失败，导致作业中断。可能是容器环境或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916907

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动测试前因自定义容器实现执行失败而中止，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916915

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装Rust工具链后，执行自定义容器实现时出错，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917352

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917389

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917442

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917444

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 8.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916930) |


## [Run #31484120985](https://github.com/sgl-project/sglang/actions/runs/31484120985)
- **分支**: `rope_mova_unification`
- **总耗时**: 34.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31484120985

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31484120985/job/93755581261) |


## [Run #31483724651](https://github.com/sgl-project/sglang/actions/runs/31483724651)
- **分支**: `agent/optimize-sm120-ernie-norm-fusion`
- **总耗时**: 7.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31483724651

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 1.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350683) |
| base-b-test-2-npu-a3 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350859) |
| base-b-test-8-npu-a3 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350878) |
| base-b-test-16-npu-a3 / run (0) | 6.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350921) |
| base-b-test-1-npu-a3 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350934) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350964) |
| base-b-test-4-npu-a3 / run (1) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351107) |
| base-b-test-4-npu-a3 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351174) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351277) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.2min | 环境问题 | CI作业因Azure Blob存储中指定文件不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351308) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351325) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351351) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350683

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350859

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350878

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350921

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350934

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350964

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351107

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源可用性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351174

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351277

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示BlobNotFound错误，说明作业依赖的某个blob文件（可能是模型权重、数据集或缓存）在存储中缺失，导致任务无法启动或运行。需检查CI配置中的blob路径或上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351308

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351325

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351351


## [Run #31483499928](https://github.com/sgl-project/sglang/actions/runs/31483499928)
- **分支**: `codex/cpu-offload-components-clean`
- **总耗时**: 28.7min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31483499928

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31483499928/job/93753600058) |


## [Run #31482217141](https://github.com/sgl-project/sglang/actions/runs/31482217141)
- **分支**: `jit-content-addressed-cache`
- **总耗时**: 67.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31482217141

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 9.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670352) |
| base-b-test-4-npu-a3 / run (1) | 10.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670392) |
| base-b-test-16-npu-a3 / run (0) | 9.6min | 代码错误 | AscendKVManager.send_kvcache()接口签名不匹配，导致传输线程崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670450) |
| base-b-test-1-npu-a3 / run (0) | 9.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670452) |
| base-b-test-2-npu-a3 / run (0) | 9.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670455) |
| base-b-test-8-npu-a3 / run (0) | 5.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670472) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.8min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670911) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670977) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749671000) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93763884763) |

- **base-b-test-4-npu-a3 / run (0)**: 日志显示服务已成功启动并处理请求，但随后出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670352

- **base-b-test-4-npu-a3 / run (1)**: 作业在加载模型权重到31%时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670392

- **base-b-test-16-npu-a3 / run (0)**: 日志显示调用send_kvcache()时传入了未预期的关键字参数'dst_kv_item_len'，而AscendKVManager的实现不支持该参数，引发TypeError并导致Prefill实例崩溃，最终作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670450

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重时失败，错误为“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670452

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型权重加载成功，但在后续执行阶段出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670455

- **base-b-test-8-npu-a3 / run (0)**: 日志显示模型加载到62%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670472

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示模型权重加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670911

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在执行过程中自定义容器实现失败，提示联系runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670977

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在11:34:00时出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749671000

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93763884763

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670275) |
| base-a-test-1-npu-a2 / run (0) | 11.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670406) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31482217141/job/93749670673) |


## [Run #31481965192](https://github.com/sgl-project/sglang/actions/runs/31481965192)
- **分支**: `main`
- **总耗时**: 45.6min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31481965192

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31481965192/job/93748934988) |


## [Run #31481725685](https://github.com/sgl-project/sglang/actions/runs/31481725685)
- **分支**: `diffusion-ideogram-rope-silu-fusion`
- **总耗时**: 58.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31481725685

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 57.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099455) |
| base-b-test-2-npu-a3 / run (0) | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099485) |
| base-b-test-4-npu-a3 / run (0) | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099517) |
| base-b-test-16-npu-a3 / run (0) | 57.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099523) |
| base-b-test-4-npu-a3 / run (1) | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099524) |
| base-b-test-1-npu-a3 / run (0) | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099586) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099677) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099682) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099736) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099737) |

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099455

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099485

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099517

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099523

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099524

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099586

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099677

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099682

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099736

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099737

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099422) |
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31481725685/job/93748099514) |


## [Run #31478363759](https://github.com/sgl-project/sglang/actions/runs/31478363759)
- **分支**: `codex/diffusion-auto-layerwise-policy`
- **总耗时**: 35.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31478363759

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31478363759/job/93737854795) |


## [Run #31477708234](https://github.com/sgl-project/sglang/actions/runs/31477708234)
- **分支**: `jit-content-addressed-cache`
- **总耗时**: 60.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31477708234

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356508) |
| base-b-test-2-npu-a3 / run (0) | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356519) |
| base-b-test-4-npu-a3 / run (1) | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356530) |
| base-b-test-1-npu-a3 / run (0) | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356617) |
| base-b-test-4-npu-a3 / run (0) | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356624) |
| base-b-test-16-npu-a3 / run (0) | 59.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356665) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735357052) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735357079) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735357083) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 59.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735357095) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356508

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356519

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356530

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356617

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356624

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该文件不存在（BlobNotFound）。这通常是日志上传失败、路径错误或文件被清理所致，属于基础设施或配置问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356665

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735357052

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735357079

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735357083

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735357095

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356260) |
| base-a-test-1-npu-a2 / run (0) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31477708234/job/93735356539) |


## [Run #31477468599](https://github.com/sgl-project/sglang/actions/runs/31477468599)
- **分支**: `codex/cpu-offload-components-clean`
- **总耗时**: 62.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31477468599

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 60.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31477468599/job/93734597746) |


## [Run #31476451719](https://github.com/sgl-project/sglang/actions/runs/31476451719)
- **分支**: `codex/dit-runtime-capabilities`
- **总耗时**: 60.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31476451719

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 53.0min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31476451719/job/93731272391) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未执行或全部通过，作业因其他原因被标记失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31476451719/job/93731272391


## [Run #31475600580](https://github.com/sgl-project/sglang/actions/runs/31475600580)
- **分支**: `main`
- **总耗时**: 77.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31475600580

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 6.4min | 代码错误 | NPU HiCache MHA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630325) |
| base-b-test-16-npu-a3 / run (0) | 76.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630328) |
| base-b-test-4-npu-a3 / run (1) | 9.7min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630340) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 超时 | NPU测试用例test_npu_hicache_mla.py执行超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630352) |
| base-b-test-8-npu-a3 / run (0) | 2.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630388) |
| base-b-test-2-npu-a3 / run (0) | 9.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630389) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630801) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630810) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630856) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 2.5min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93746251346) |

- **base-b-test-1-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mha.py 执行失败，0/11 测试通过，耗时159秒，具体错误信息未在日志中显示，需查看详细测试输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630325

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是因为日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630328

- **base-b-test-4-npu-a3 / run (1)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630340

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test_npu_hicache_mla.py运行301秒后超时退出，返回码1，导致整个作业失败。该测试属于HiCache功能测试，可能因NPU资源或代码问题导致执行时间超过预期。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630352

- **base-b-test-8-npu-a3 / run (0)**: 日志显示在安装CANN自定义算子后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630388

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试已正常运行并输出结果，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630389

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示模型权重加载到65%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630801

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中服务正常响应，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630810

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630856

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93746251346

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 45.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630283) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630508) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475600580/job/93728630759) |


## [Run #31475446607](https://github.com/sgl-project/sglang/actions/runs/31475446607)
- **分支**: `codex/cpu-offload-components-clean`
- **总耗时**: 27.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31475446607

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475446607/job/93728104663) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业整体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475446607/job/93728104663


## [Run #31475201763](https://github.com/sgl-project/sglang/actions/runs/31475201763)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 179.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31475201763

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 64.3min | 代码错误 | NPU PD分离测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319088) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.5min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93745341278) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现同PR中另一作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93753479480) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93754321870) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93764768006) |

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（exit code 1），耗时346秒，其余3个测试均通过。可能是PD分离功能相关代码或测试用例本身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319088

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时约1060秒，未通过性能基准，可能因模型推理速度或延迟不满足50ms要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93745341278

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，将其判定为根因作业，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93753479480

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在启动阶段即被健康检查拦截，原因是同PR中base-c-test-perf-8-npu-a3作业已失败，触发了快速失败机制，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93754321870

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业（base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3），触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93764768006

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 47.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727318935) |
| base-b-test-4-npu-a3 / run (1) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319025) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319035) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319065) |
| base-b-test-2-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319070) |
| base-b-test-4-npu-a3 / run (0) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319084) |
| base-b-test-1-npu-a3 / run (0) | 26.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319109) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319308) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319311) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319361) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31475201763/job/93727319393) |


## [Run #31474177768](https://github.com/sgl-project/sglang/actions/runs/31474177768)
- **分支**: `codex/kimi-k3-npu-main-20260803`
- **总耗时**: 27.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31474177768

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境警告和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076570) |
| base-b-test-16-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076684) |
| base-b-test-8-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076725) |
| base-b-test-4-npu-a3 / run (1) | 26.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076775) |
| base-b-test-2-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076779) |
| base-b-test-1-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076798) |
| base-b-test-4-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076890) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724077153) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724077232) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724077308) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724077343) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示Node.js 20弃用警告和上传diffusion-failures产物时未找到文件，说明测试可能未产生失败样本，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076570

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076684

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076725

- **base-b-test-4-npu-a3 / run (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076775

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该文件不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076779

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关依赖资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076798

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076890

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724077153

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个远程文件（如模型权重、测试数据或缓存）已被删除或路径错误，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724077232

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重、测试数据或构建产物）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724077308

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724077343

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31474177768/job/93724076687) |


## [Run #31474095472](https://github.com/sgl-project/sglang/actions/runs/31474095472)
- **分支**: `main`
- **总耗时**: 20.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31474095472

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 17.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779716) |
| base-b-test-16-npu-a3 / run (0) | 19.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779827) |
| base-b-test-2-npu-a3 / run (0) | 19.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779904) |
| base-b-test-4-npu-a3 / run (0) | 19.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779947) |
| base-b-test-1-npu-a3 / run (0) | 19.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779949) |
| base-b-test-4-npu-a3 / run (1) | 19.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779963) |
| base-a-test-1-npu-a2 / run (0) | 1.3min | 环境问题 | 自定义容器执行失败，健康检查阶段报错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779967) |
| base-b-test-8-npu-a3 / run (0) | 19.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780004) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780311) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780403) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780451) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780466) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。无法判断具体失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779716

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779827

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779904

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是构建产物或依赖缓存缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779947

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779949

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关依赖资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779963

- **base-a-test-1-npu-a2 / run (0)**: 作业在health-check阶段执行自定义容器时失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723779967

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780004

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780311

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780403

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780451

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31474095472/job/93723780466


---
*Auto-generated by npu_pr_monitor.py*