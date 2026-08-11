# NPU CI 执行监控
**生成时间**: 2026-08-11 00:18 UTC
**分析 Run 数**: 20

---

## [Run #31443648815](https://github.com/sgl-project/sglang/actions/runs/31443648815)
- **分支**: `main`
- **总耗时**: 10.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31443648815

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405495) |
| multimodal-gen-test-2-npu-a3 | 9.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405500) |
| base-b-test-16-npu-a3 / run (0) | 9.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405585) |
| base-b-test-2-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405599) |
| base-b-test-8-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405625) |
| base-a-test-1-npu-a2 / run (0) | 9.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405630) |
| base-b-test-4-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405652) |
| base-b-test-1-npu-a3 / run (0) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405661) |
| base-b-test-4-npu-a3 / run (1) | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405662) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405884) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405909) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405933) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633406018) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未捕获到multimodal-gen测试的具体执行日志或错误输出，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405495

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（未找到失败文件），未展示multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405500

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是缓存、依赖或日志文件缺失，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405585

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问 Azure Blob 中的日志文件，但该文件已被删除或路径错误，返回 BlobNotFound 错误。这属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405599

- **base-b-test-8-npu-a3 / run (0)**: 作业运行时尝试下载或访问的 blob 资源（如日志文件）已被删除或路径错误，返回 BlobNotFound 错误。这属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405625

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405630

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405652

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405661

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405662

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405884

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405909

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633405933

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443648815/job/93633406018


## [Run #31443022727](https://github.com/sgl-project/sglang/actions/runs/31443022727)
- **分支**: `main`
- **总耗时**: 10.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31443022727

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 9.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561404) |
| multimodal-gen-test-1-npu-a3 | 9.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561474) |
| base-b-test-2-npu-a3 / run (0) | 9.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561482) |
| base-b-test-4-npu-a3 / run (1) | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561496) |
| base-b-test-4-npu-a3 / run (0) | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561511) |
| base-a-test-1-npu-a2 / run (0) | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561584) |
| base-b-test-8-npu-a3 / run (0) | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561590) |
| base-b-test-1-npu-a3 / run (0) | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561606) |
| base-b-test-16-npu-a3 / run (0) | 9.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561609) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561892) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561933) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561934) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561963) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561404

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤。可能因日志截断或作业在测试前已失败，需查看完整日志以确定根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561474

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561482

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561496

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561511

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561584

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的工件/文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561590

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561606

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明作业尝试访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561609

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561892

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561933

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561934

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置变更导致，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31443022727/job/93631561963


## [Run #31441135349](https://github.com/sgl-project/sglang/actions/runs/31441135349)
- **分支**: `main`
- **总耗时**: 19.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31441135349

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 18.4min | 环境问题 | 作业因未找到失败产物文件而提前结束，实际测试结果未生成。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055506) |
| multimodal-gen-test-1-npu-a3 | 18.1min | 环境问题 | 作业因环境问题失败，未找到diffusion-failures目录，测试未执行或提前退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055541) |
| base-b-test-4-npu-a3 / run (1) | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055561) |
| base-a-test-1-npu-a2 / run (0) | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055583) |
| base-b-test-4-npu-a3 / run (0) | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055586) |
| base-b-test-2-npu-a3 / run (0) | 18.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055596) |
| base-b-test-8-npu-a3 / run (0) | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055606) |
| base-b-test-1-npu-a3 / run (0) | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055622) |
| base-b-test-16-npu-a3 / run (0) | 18.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055674) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055871) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055896) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055922) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 18.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055933) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures目录时提示无文件，说明测试未产生失败样本，可能因环境配置或测试逻辑问题导致作业未正常执行完成。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055506

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures工件时提示无文件，说明测试未生成失败样本，可能因NPU环境配置或依赖问题导致测试未正常运行，需检查环境初始化步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055541

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055561

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055583

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是缓存或依赖文件未正确上传，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055586

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055596

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055606

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的工件/缓存文件已被删除或路径错误，属于外部存储依赖问题，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055622

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055674

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055871

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055896

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055922

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441135349/job/93626055933


## [Run #31441098686](https://github.com/sgl-project/sglang/actions/runs/31441098686)
- **分支**: `cheng/unified-memory-pd-disagg`
- **总耗时**: 6.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31441098686

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 5.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901612) |
| base-b-test-1-npu-a3 / run (0) | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901791) |
| base-b-test-2-npu-a3 / run (0) | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901834) |
| base-b-test-4-npu-a3 / run (0) | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901852) |
| base-b-test-8-npu-a3 / run (0) | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901854) |
| base-a-test-1-npu-a2 / run (0) | 0.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901890) |
| base-b-test-4-npu-a3 / run (1) | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902029) |
| base-b-test-16-npu-a3 / run (0) | 0.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902093) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902325) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902389) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902435) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902483) |

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901612

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、缓存或构建产物）在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置不一致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901791

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901834

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901852

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是上传失败或路径错误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901854

- **base-a-test-1-npu-a2 / run (0)**: 作业运行时尝试下载或访问 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625901890

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902029

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902093

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902325

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902389

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902435

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31441098686/job/93625902483


## [Run #31440438469](https://github.com/sgl-project/sglang/actions/runs/31440438469)
- **分支**: `cheng/gc-sr-review`
- **总耗时**: 64.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31440438469

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 33.3min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948707) |
| multimodal-gen-test-2-npu-a3 | 36.8min | 其他 | 作业未显示实际测试失败，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948773) |
| base-b-test-1-npu-a3 / run (0) | 63.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948856) |
| base-b-test-8-npu-a3 / run (0) | 63.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948935) |
| base-b-test-16-npu-a3 / run (0) | 63.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948953) |
| base-b-test-2-npu-a3 / run (0) | 63.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948979) |
| base-b-test-4-npu-a3 / run (1) | 63.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948986) |
| base-b-test-4-npu-a3 / run (0) | 63.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948990) |
| base-a-test-1-npu-a2 / run (0) | 63.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948995) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 63.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623949339) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 63.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623949467) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 63.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623949471) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 63.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623949499) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、Node版本警告及上传diffusion-failures工件（未找到文件）等常规信息，未包含multimodal-gen-test实际执行和失败的具体错误，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948707

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures目录时提示无文件，未发现测试执行或失败信息，可能测试未运行或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948773

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948856

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象缺失，可能是构建产物未上传、路径错误或存储被清理，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948935

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948953

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948979

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于环境配置或资源缺失问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948986

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948990

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623948995

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623949339

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623949467

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623949471

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31440438469/job/93623949499


## [Run #31438996490](https://github.com/sgl-project/sglang/actions/runs/31438996490)
- **分支**: `main`
- **总耗时**: 10.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31438996490

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548827) |
| multimodal-gen-test-2-npu-a3 | 9.7min | 环境问题 | 作业因未找到diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548854) |
| base-a-test-1-npu-a2 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548922) |
| base-b-test-16-npu-a3 / run (0) | 9.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548930) |
| base-b-test-4-npu-a3 / run (1) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548939) |
| base-b-test-1-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548942) |
| base-b-test-2-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548989) |
| base-b-test-4-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548996) |
| base-b-test-8-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549066) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549246) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549292) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549324) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549357) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548827

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures工件时提示无文件，说明测试未生成失败样本，可能因环境配置或前置步骤异常导致测试未运行或全部跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548854

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548922

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548930

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548939

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548942

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548989

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619548996

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549066

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549246

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549292

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549324

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31438996490/job/93619549357


## [Run #31437565937](https://github.com/sgl-project/sglang/actions/runs/31437565937)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 5.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31437565937

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 4.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163856) |
| base-b-test-8-npu-a3 / run (0) | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163909) |
| multimodal-gen-test-1-npu-a3 | 4.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163911) |
| base-b-test-4-npu-a3 / run (1) | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163946) |
| base-b-test-16-npu-a3 / run (0) | 4.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163958) |
| base-a-test-1-npu-a2 / run (0) | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163960) |
| base-b-test-1-npu-a3 / run (0) | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163963) |
| base-b-test-2-npu-a3 / run (0) | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163971) |
| base-b-test-4-npu-a3 / run (0) | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164004) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164177) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164197) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.8min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164210) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164223) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败断言信息，仅显示上传diffusion-failures工件时无文件，可能测试未运行或日志被截断，需查看完整日志定位失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163856

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163909

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但实际失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163911

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163946

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163958

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163960

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163963

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615163971

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164004

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164177

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164197

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示BlobNotFound错误，说明CI作业尝试下载的依赖文件或缓存未在存储中找到，可能是资源被清理或路径配置错误，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164210

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437565937/job/93615164223


## [Run #31437157417](https://github.com/sgl-project/sglang/actions/runs/31437157417)
- **分支**: `muse-glimmer`
- **总耗时**: 97.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31437157417

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 31.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613833884) |
| base-b-test-8-npu-a3 / run (0) | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834005) |
| multimodal-gen-test-2-npu-a3 | 20.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834006) |
| base-b-test-1-npu-a3 / run (0) | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834034) |
| base-b-test-4-npu-a3 / run (1) | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834072) |
| base-b-test-2-npu-a3 / run (0) | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834078) |
| base-b-test-4-npu-a3 / run (0) | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834082) |
| base-a-test-1-npu-a2 / run (0) | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834091) |
| base-b-test-16-npu-a3 / run (0) | 97.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834110) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834357) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834359) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834405) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 97.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834422) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613833884

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件已被删除或路径错误，属于基础设施/存储配置问题，需检查相关 blob 是否存在或更新下载链接。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834005

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未展示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834006

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834034

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834072

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834078

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是缓存或依赖文件未正确上传，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834082

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径和生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834091

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834110

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834357

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）未上传或已被删除，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834359

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834405

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31437157417/job/93613834422


## [Run #31434156233](https://github.com/sgl-project/sglang/actions/runs/31434156233)
- **分支**: `main`
- **总耗时**: 39.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31434156233

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 39.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488095) |
| base-b-test-8-npu-a3 / run (0) | 39.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488099) |
| base-b-test-16-npu-a3 / run (0) | 39.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488107) |
| base-a-test-1-npu-a2 / run (0) | 39.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488138) |
| base-b-test-1-npu-a3 / run (0) | 39.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488149) |
| base-b-test-4-npu-a3 / run (1) | 39.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488160) |
| base-b-test-4-npu-a3 / run (0) | 39.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488183) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 39.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488392) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488496) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 39.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488500) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 39.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488568) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488095

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488099

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、文件被清理或配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488107

- **base-a-test-1-npu-a2 / run (0)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488138

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488149

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488160

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob资源已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488183

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488392

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径配置错误，需检查存储账户和 blob 路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488496

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488500

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储中找到，可能是资源被清理、路径错误或上传失败，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31434156233/job/93604488568


## [Run #31433764898](https://github.com/sgl-project/sglang/actions/runs/31433764898)
- **分支**: `feat/add-cosmos3-edge-distil`
- **总耗时**: 26.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31433764898

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31433764898/job/93603215659) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含任何测试执行或失败的具体信息，只有GitHub Actions的初始化、上传artifact（无文件）和清理步骤。可能测试在中间被截断或未运行，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31433764898/job/93603215659

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31433764898/job/93603215692) |


## [Run #31429924525](https://github.com/sgl-project/sglang/actions/runs/31429924525)
- **分支**: `feat/add-cosmos3-edge-distil`
- **总耗时**: 28.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31429924525

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429924525/job/93590688096) |

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429924525/job/93590688096

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31429924525/job/93590688187) |


## [Run #31429438178](https://github.com/sgl-project/sglang/actions/runs/31429438178)
- **分支**: `main`
- **总耗时**: 59.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31429438178

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589152805) |
| base-b-test-1-npu-a3 / run (0) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589152869) |
| base-b-test-16-npu-a3 / run (0) | 58.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589152907) |
| base-a-test-1-npu-a2 / run (0) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589152984) |
| base-b-test-4-npu-a3 / run (0) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153016) |
| base-b-test-4-npu-a3 / run (1) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153031) |
| base-b-test-8-npu-a3 / run (0) | 58.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153182) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153918) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153924) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153982) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589154005) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589152805

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589152869

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是由于文件被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589152907

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589152984

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153016

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存或依赖文件未正确上传，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153031

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153182

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源未上传或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153918

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是上传失败、过期或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153924

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589153982

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429438178/job/93589154005


## [Run #31429418511](https://github.com/sgl-project/sglang/actions/runs/31429418511)
- **分支**: `perf/hisparse-dsv4-fused-swap-in`
- **总耗时**: 59.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31429418511

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 58.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046807) |
| base-b-test-4-npu-a3 / run (0) | 58.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046870) |
| base-b-test-1-npu-a3 / run (0) | 58.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046875) |
| base-b-test-2-npu-a3 / run (0) | 58.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046877) |
| base-a-test-1-npu-a2 / run (0) | 58.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046911) |
| base-b-test-8-npu-a3 / run (0) | 58.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046912) |
| base-b-test-4-npu-a3 / run (1) | 58.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046923) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 58.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589047220) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 58.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589047228) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 58.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589047235) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 58.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589047289) |

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046807

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径配置错误，或资源已被删除。属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046870

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源（如模型权重、缓存或日志）已被删除或路径错误，属于环境或配置问题，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046875

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046877

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046911

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046912

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存、依赖或上传文件未正确生成，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589046923

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589047220

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589047228

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589047235

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429418511/job/93589047289


## [Run #31429130441](https://github.com/sgl-project/sglang/actions/runs/31429130441)
- **分支**: `muse-glimmer`
- **总耗时**: 103.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31429130441

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 19.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251518) |
| base-b-test-4-npu-a3 / run (0) | 101.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251765) |
| base-b-test-16-npu-a3 / run (0) | 101.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251780) |
| base-b-test-1-npu-a3 / run (0) | 101.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251846) |
| base-b-test-4-npu-a3 / run (1) | 101.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251877) |
| base-b-test-8-npu-a3 / run (0) | 101.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251898) |
| base-a-test-1-npu-a2 / run (0) | 101.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252048) |
| base-b-test-2-npu-a3 / run (0) | 101.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252083) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 101.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252199) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 101.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252224) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 101.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252245) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 101.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252272) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251518

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251765

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/日志文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251780

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251846

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251877

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251898

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252048

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252083

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252199

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252224

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或被误删，需检查存储配置及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252245

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588252272

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31429130441/job/93588251627) |


## [Run #31428638383](https://github.com/sgl-project/sglang/actions/runs/31428638383)
- **分支**: `main`
- **总耗时**: 6.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31428638383

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409107) |
| multimodal-gen-test-2-npu-a3 | 5.2min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409114) |
| base-b-test-1-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409311) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409326) |
| base-b-test-8-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409344) |
| base-b-test-2-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409388) |
| base-b-test-4-npu-a3 / run (1) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409419) |
| base-b-test-16-npu-a3 / run (0) | 5.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409430) |
| base-b-test-4-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409567) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409917) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409930) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409996) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409998) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅有GitHub Actions环境准备、Node版本警告及上传失败产物（无文件）等常规信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409107

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未包含multimodal-gen-test-2-npu-a3作业的实际测试命令或错误输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409114

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409311

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409326

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409344

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409388

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409419

- **base-b-test-16-npu-a3 / run (0)**: 作业在下载或访问某个blob资源时，返回BlobNotFound错误，说明该资源已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409430

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409567

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409917

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409930

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409996

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31428638383/job/93586409998


## [Run #31425204700](https://github.com/sgl-project/sglang/actions/runs/31425204700)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 156.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31425204700

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320437) |
| base-b-test-16-npu-a3 / run (0) | 155.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320618) |
| base-b-test-1-npu-a3 / run (0) | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320721) |
| base-a-test-1-npu-a2 / run (0) | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320764) |
| base-b-test-2-npu-a3 / run (0) | 155.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320774) |
| base-b-test-8-npu-a3 / run (0) | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320778) |
| base-b-test-4-npu-a3 / run (1) | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320820) |
| base-b-test-4-npu-a3 / run (0) | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320844) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575321232) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575321269) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575321275) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 155.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575321333) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载和artifact上传（无文件）。无法判断失败原因，可能是测试未运行或日志被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320437

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320618

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物未上传或路径错误，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320721

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320764

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320774

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320778

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源（可能是预构建镜像或依赖文件）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320820

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320844

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储账户中找到，可能是资源被清理、路径错误或上传失败，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575321232

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575321269

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575321275

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575321333

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31425204700/job/93575320556) |


## [Run #31424514186](https://github.com/sgl-project/sglang/actions/runs/31424514186)
- **分支**: `kda-helion-backend`
- **总耗时**: 102.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31424514186

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 22.3min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113209) |
| base-b-test-1-npu-a3 / run (0) | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113324) |
| base-a-test-1-npu-a2 / run (0) | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113390) |
| base-b-test-2-npu-a3 / run (0) | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113396) |
| base-b-test-4-npu-a3 / run (0) | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113427) |
| base-b-test-4-npu-a3 / run (1) | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113452) |
| base-b-test-16-npu-a3 / run (0) | 101.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113458) |
| base-b-test-8-npu-a3 / run (0) | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113476) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113823) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113853) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113887) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 101.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113966) |

- **multimodal-gen-test-2-npu-a3**: 日志显示上传diffusion-failures目录时无文件，但未出现测试失败或错误信息，可能为测试通过但产物缺失，或日志截断导致关键错误未显示。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113209

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113324

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是上游产物未上传或过期，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113390

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113396

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113427

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113452

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113458

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113476

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113823

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113853

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113887

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113966

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31424514186/job/93573113174) |


## [Run #31423052373](https://github.com/sgl-project/sglang/actions/runs/31423052373)
- **分支**: `wip/gdn-wy-verify-cudagraph`
- **总耗时**: 204.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31423052373

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335181) |
| base-b-test-8-npu-a3 / run (0) | 203.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335327) |
| base-b-test-1-npu-a3 / run (0) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335350) |
| base-b-test-4-npu-a3 / run (0) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335361) |
| base-b-test-4-npu-a3 / run (1) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335362) |
| base-b-test-16-npu-a3 / run (0) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335411) |
| base-a-test-1-npu-a2 / run (0) | 203.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335414) |
| base-b-test-2-npu-a3 / run (0) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335457) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335744) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335750) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335798) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335819) |

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335181

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335327

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存失效或资源清理导致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335350

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335361

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335362

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335411

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中已被删除或路径错误，属于基础设施配置问题，需检查上传步骤或存储生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335414

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335457

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335744

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335750

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335798

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335819

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31423052373/job/93568335205) |


## [Run #31422215072](https://github.com/sgl-project/sglang/actions/runs/31422215072)
- **分支**: `amd/mm-gen-nested-b2-envskip`
- **总耗时**: 27.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31422215072

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31422215072/job/93565544538) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或已通过，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31422215072/job/93565544538

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31422215072/job/93565544366) |


## [Run #31421197932](https://github.com/sgl-project/sglang/actions/runs/31421197932)
- **分支**: `diffusion-ltx2-modulate-mount`
- **总耗时**: 33.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31421197932

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.2min | 其他 | 作业未显示明确失败原因，仅上传artifact时提示无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31421197932/job/93562248139) |

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试失败或错误信息，仅显示upload-artifact因无diffusion-failures文件而跳过，作业可能因测试通过但无失败产物而正常结束，或日志被截断未显示关键错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31421197932/job/93562248139

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31421197932/job/93562247960) |


---
*Auto-generated by npu_pr_monitor.py*