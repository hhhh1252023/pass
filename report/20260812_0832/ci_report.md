# NPU CI 执行监控
**生成时间**: 2026-08-12 00:32 UTC
**分析 Run 数**: 21

---

## [Run #31546807108](https://github.com/sgl-project/sglang/actions/runs/31546807108)
- **分支**: `main`
- **总耗时**: 15.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31546807108

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014272) |
| base-b-test-8-npu-a3 / run (0) | 11.3min | 其他 | 作业实际成功，无失败原因 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014432) |
| base-b-test-16-npu-a3 / run (0) | 13.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014539) |
| base-b-test-1-npu-a3 / run (0) | 11.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014541) |
| base-b-test-4-npu-a3 / run (0) | 8.3min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014556) |
| base-b-test-2-npu-a3 / run (0) | 11.2min | 环境问题 | 自定义容器执行失败，runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014601) |
| base-b-test-4-npu-a3 / run (1) | 11.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014680) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014865) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014873) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014904) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 7.2min | 环境问题 | 自定义容器执行失败，模型权重加载过程中容器异常退出 | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93962428754) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误，仅有Node 20弃用警告和diffusion-failures目录无文件上传提示，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014272

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试全部通过（1/1 passed），作业正常结束，仅包含Node.js 20弃用警告，无实际错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014432

- **base-b-test-16-npu-a3 / run (0)**: 日志显示服务已成功启动并完成预热，但随后报错“Executing the custom container implementation failed”，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014539

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中突然报错'Executing the custom container implementation failed'，提示联系runner管理员，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014541

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试执行失败，退出码为1，耗时281秒，导致整个作业失败。具体失败原因需查看该测试文件的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014556

- **base-b-test-2-npu-a3 / run (0)**: 测试在运行第2个测试文件时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于runner环境或容器配置问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014601

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行正常，但随后出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014680

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner容器环境问题而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014865

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014873

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常处理请求，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境或容器执行问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014904

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在加载模型shards到79%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93962428754

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014455) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31546807108/job/93961014915) |


## [Run #31545706220](https://github.com/sgl-project/sglang/actions/runs/31545706220)
- **分支**: `lsyin/dsv4-parser-fix`
- **总耗时**: 13.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31545706220

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739357) |
| base-b-test-1-npu-a3 / run (0) | 10.6min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739505) |
| base-b-test-2-npu-a3 / run (0) | 10.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739510) |
| base-b-test-4-npu-a3 / run (1) | 10.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739536) |
| base-b-test-4-npu-a3 / run (0) | 10.4min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739615) |
| base-b-test-16-npu-a3 / run (0) | 8.0min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739717) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957740358) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957740362) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957740396) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 4.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93959227488) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。无法判断具体失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739357

- **base-b-test-1-npu-a3 / run (0)**: 测试运行到第4个文件时，自定义容器实现执行失败，报错提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739505

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在模型加载和tokenizer初始化后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739510

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型权重加载成功（Qwen3MoeForCausalLM），但在后续执行时出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739536

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试正在正常运行（TestDPAttentionDP2TP2.test_regex_generate_phone），但突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739615

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动NPU容器后，Watchdog TokenizerManager初始化正常，但随后自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739717

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957740358

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（HTTP 200），但随后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner或容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957740362

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在加载模型分片过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957740396

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU CI环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93959227488

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739611) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957739704) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31545706220/job/93957740184) |


## [Run #31545315405](https://github.com/sgl-project/sglang/actions/runs/31545315405)
- **分支**: `lsyin/dsv4-parser-fix`
- **总耗时**: 6.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31545315405

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 4.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612693) |
| multimodal-gen-test-1-npu-a3 | 4.3min | 环境问题 | 作业因环境问题失败，未找到失败产物文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612702) |
| base-b-test-16-npu-a3 / run (0) | 3.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612740) |
| base-b-test-2-npu-a3 / run (0) | 4.6min | 环境问题 | 自托管runner执行自定义容器时失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612743) |
| base-a-test-1-npu-a2 / run (0) | 3.8min | 环境问题 | 自定义容器执行失败，导致测试未运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612758) |
| base-b-test-8-npu-a3 / run (0) | 2.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612826) |
| base-b-test-4-npu-a3 / run (0) | 4.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612931) |
| base-b-test-4-npu-a3 / run (1) | 4.0min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612954) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.6min | 环境问题 | 自定义容器执行失败，下载triton-ascend依赖时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956613164) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956613215) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.8min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956613252) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.8min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956613269) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612693

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在运行后上传diffusion-failures产物时提示无文件，说明测试未生成失败记录，可能因环境配置或依赖问题导致测试未正常执行，而非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612702

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动自定义容器后，TokenizerManager初始化过程中容器执行失败，报错提示联系自托管runner管理员，属于NPU测试环境基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612740

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型权重加载到25%时，runner报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612743

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在运行测试前，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612758

- **base-b-test-8-npu-a3 / run (0)**: 作业在安装依赖后执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，可能与NPU驱动或容器配置有关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612826

- **base-b-test-4-npu-a3 / run (0)**: 日志显示容器启动后初始化TokenizerManager时失败，报错'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612931

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动NPU容器后，Tokenizer初始化阶段出现自定义容器实现执行失败错误，可能是容器镜像或NPU驱动环境配置问题，导致测试无法继续运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956612954

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业提前终止，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956613164

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于runner或容器环境配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956613215

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在设置环境变量后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境或容器配置问题，而非测试代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956613252

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31545315405/job/93956613269


## [Run #31544686857](https://github.com/sgl-project/sglang/actions/runs/31544686857)
- **分支**: `lsyin/dsv4-parser-fix`
- **总耗时**: 9.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31544686857

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.2min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695106) |
| base-b-test-8-npu-a3 / run (0) | 3.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695208) |
| base-b-test-4-npu-a3 / run (0) | 2.5min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695346) |
| base-b-test-4-npu-a3 / run (1) | 3.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695351) |
| base-b-test-1-npu-a3 / run (0) | 8.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695379) |
| base-b-test-2-npu-a3 / run (0) | 4.6min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695404) |
| base-b-test-16-npu-a3 / run (0) | 1.9min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695408) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.0min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695642) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 环境问题 | 作业在准备阶段即失败，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695805) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.0min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695834) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.0min | 环境问题 | 作业在准备阶段被中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93956234885) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行测试后上传diffusion-failures目录时提示无文件，但日志中间部分被省略，无法判断是测试通过但误报失败，还是测试失败但未生成产物。需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695106

- **base-b-test-8-npu-a3 / run (0)**: 作业在运行测试前，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695208

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在构建sglang包时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是容器镜像或运行环境配置问题，而非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695346

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695351

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695379

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型权重加载到50%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695404

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于环境配置或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695408

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型权重加载正常，但在设置ASCEND_OPP_PATH后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695642

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示作业在checkout后运行自定义脚本'/home/runner/k8s/index.js'时中断，无后续测试输出，疑似runner环境或基础设施问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695805

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在启动测试容器后，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695834

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示作业在checkout后运行k8s/index.js时被清理，无测试输出，可能因runner环境或调度问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93956234885

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695389) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31544686857/job/93954695815) |


## [Run #31544345808](https://github.com/sgl-project/sglang/actions/runs/31544345808)
- **分支**: `main`
- **总耗时**: 18.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31544345808

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.8min | 其他 | 作业日志不完整，未显示测试执行过程，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650098) |
| base-b-test-16-npu-a3 / run (0) | 12.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650172) |
| base-b-test-2-npu-a3 / run (0) | 10.6min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650197) |
| base-b-test-4-npu-a3 / run (0) | 7.9min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650215) |
| base-b-test-1-npu-a3 / run (0) | 12.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650252) |
| base-b-test-4-npu-a3 / run (1) | 11.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650262) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.6min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650416) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650450) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650479) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93955487331) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions的初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试命令或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650098

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650172

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型分片加载到75%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650197

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试执行失败（exit code 1），耗时281秒，导致整个作业失败。具体错误信息未在日志中显示，需查看该测试的详细输出以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650215

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650252

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行正常，但在23:12:21时出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650262

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示模型分片加载到62%时，GitHub Actions报错“Executing the custom container implementation failed”，属于自托管runner容器环境异常，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650416

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在执行过程中出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650450

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常（gsm8k评估进行中），但突然报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner容器环境故障，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650479

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段因自定义容器实现执行失败而终止，错误提示联系自托管runner管理员，属于runner或容器环境配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93955487331

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650156) |
| base-b-test-8-npu-a3 / run (0) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650212) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31544345808/job/93953650537) |


## [Run #31544296716](https://github.com/sgl-project/sglang/actions/runs/31544296716)
- **分支**: `lsyin/dsv4-parser-fix`
- **总耗时**: 6.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31544296716

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582579) |
| base-b-test-8-npu-a3 / run (0) | 4.0min | 环境问题 | 自定义容器执行失败，NPU测试环境未就绪 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582613) |
| base-b-test-4-npu-a3 / run (1) | 1.7min | 环境问题 | 自定义容器执行失败，安装torch-npu时出错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582716) |
| base-a-test-1-npu-a2 / run (0) | 4.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582769) |
| base-b-test-1-npu-a3 / run (0) | 3.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582800) |
| base-b-test-4-npu-a3 / run (0) | 0.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582848) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 2.8min | 环境问题 | 自定义容器执行失败，导致作业在启动阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953583061) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.0min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953583121) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 0.7min | 环境问题 | 自定义容器启动失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953583191) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行测试后上传diffusion-failures目录时提示无文件，但日志中间部分被省略，无法判断具体失败点。可能是测试未执行或全部通过，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582579

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动测试前，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，与测试代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582613

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在安装torch-npu包时，执行自定义容器实现失败，错误为'Executing the custom container implementation failed'，可能是容器环境或依赖问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582716

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行测试前，执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582769

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在运行测试时出现'Executing the custom container implementation failed'错误，提示联系self-hosted runner管理员，属于NPU容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582800

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动阶段执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于基础设施或容器配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953582848

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于运行环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953583061

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在初始化GLM-4.7-Flash模型后，出现'Executing the custom container implementation failed'错误，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953583121

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动自定义容器时失败，错误提示为执行自定义容器实现失败，需联系自托管runner管理员。日志显示runner版本正常，但容器启动阶段报错，属于环境配置或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953583191

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31544296716/job/93953583272) |


## [Run #31543510054](https://github.com/sgl-project/sglang/actions/runs/31543510054)
- **分支**: `main`
- **总耗时**: 12.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31543510054

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 7.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951108971) |
| multimodal-gen-test-1-npu-a3 | 11.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109038) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | 环境问题 | 自定义容器执行失败，测试本身通过但作业因容器问题终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109074) |
| base-b-test-1-npu-a3 / run (0) | 10.9min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109076) |
| base-b-test-4-npu-a3 / run (0) | 8.0min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109109) |
| base-b-test-4-npu-a3 / run (1) | 8.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109184) |
| base-b-test-2-npu-a3 / run (0) | 10.6min | 环境问题 | 自定义容器执行失败，NPU测试环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109201) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.6min | 环境问题 | 自定义容器执行失败，导致作业在安装依赖后崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109411) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109587) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109596) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 3.5min | 环境问题 | 自定义容器执行失败，导致测试未开始即终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93952701873) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试进行到88%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于NPU CI环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951108971

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109038

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试用例全部通过（200/200，ok），但随后出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109074

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常（HTTP 200），但随后出现"Executing the custom container implementation failed"错误，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109076

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 执行失败，耗时281秒，0/5测试通过。可能是测试代码逻辑错误或NPU环境兼容性问题，需查看具体测试输出定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109109

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行到92%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109184

- **base-b-test-2-npu-a3 / run (0)**: 日志显示服务启动成功但随后报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109201

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在pip安装evalscope等依赖成功后，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109411

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109587

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示模型权重加载到89%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109596

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试命令刚启动（约7秒后）即报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93952701873

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109253) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31543510054/job/93951109474) |


## [Run #31542662765](https://github.com/sgl-project/sglang/actions/runs/31542662765)
- **分支**: `lsyin/dsv4-parser-fix`
- **总耗时**: 20.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31542662765

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530699) |
| base-b-test-16-npu-a3 / run (0) | 18.4min | 环境问题 | NPU容器执行失败，模型权重加载时发生崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530795) |
| base-b-test-1-npu-a3 / run (0) | 18.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530918) |
| base-b-test-4-npu-a3 / run (0) | 16.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530949) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948531294) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948531377) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 18.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948531439) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 14.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93949564092) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的输出，仅有GitHub Actions基础设施信息（Node版本警告、上传artifact等），无法判断具体失败原因，可能为日志截断或作业在测试前已异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530699

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载MoE模型权重时，torch::autograd::copy_操作崩溃，随后Scheduler watchdog超时，最终自定义容器执行失败。可能是NPU环境或容器配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530795

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530918

- **base-b-test-4-npu-a3 / run (0)**: 日志显示容器运行中突然报错“Executing the custom container implementation failed”，随后进入清理流程，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530949

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量稳定），但突然报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948531294

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但执行自定义容器时失败，提示联系runner管理员，属于runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948531377

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在22:50:03时出现"Executing the custom container implementation failed"错误，可能是自托管runner环境问题，导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948531439

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93949564092

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530856) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530859) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530878) |
| base-b-test-4-npu-a3 / run (1) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948530953) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542662765/job/93948531415) |


## [Run #31542497629](https://github.com/sgl-project/sglang/actions/runs/31542497629)
- **分支**: `main`
- **总耗时**: 14.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31542497629

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948035939) |
| base-b-test-2-npu-a3 / run (0) | 12.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036089) |
| base-b-test-4-npu-a3 / run (1) | 13.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036102) |
| base-b-test-4-npu-a3 / run (0) | 8.0min | 代码错误 | HiCache MLA 测试用例执行失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036138) |
| base-b-test-1-npu-a3 / run (0) | 12.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036187) |
| base-b-test-16-npu-a3 / run (0) | 12.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036208) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036475) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036552) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.3min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036750) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93949082565) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948035939

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在模型加载和tokenizer初始化后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036089

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中服务正常响应，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036102

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 在运行约281秒后失败，0/5测试通过，具体断言或运行错误需查看该测试的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036138

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中容器突然报错"Executing the custom container implementation failed"，提示联系runner管理员，属于NPU自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036187

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员。日志显示NPU内存正常（约61GB），但容器环境本身存在问题，导致测试无法继续。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036208

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036475

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在评估gsm8k过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036552

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示模型权重加载到85%时，runner报错'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036750

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在NPU图捕获阶段（bs=20）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93949082565

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 10.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036107) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036170) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542497629/job/93948036615) |


## [Run #31542031058](https://github.com/sgl-project/sglang/actions/runs/31542031058)
- **分支**: `kv-budget-mm-reservation`
- **总耗时**: 51.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31542031058

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 49.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482484) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 46.9min | 环境问题 | 自定义容器执行失败，runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946483014) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93948022732) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 25.0min | 环境问题 | 自定义容器执行失败，导致性能测试中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93952005524) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 6.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93955861922) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型分片（约44%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482484

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试正常运行中，但突然出现"Executing the custom container implementation failed"错误，提示联系self-hosted runner管理员，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946483014

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1136秒，未通过性能基准要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93948022732

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示bench_serving命令已启动，但随后出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93952005524

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示模型权重加载完成后，GitHub Actions 报错“Executing the custom container implementation failed”，属于自托管运行器容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93955861922

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482333) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482425) |
| base-b-test-2-npu-a3 / run (0) | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482431) |
| base-b-test-4-npu-a3 / run (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482457) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482495) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482528) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482666) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482971) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946482987) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31542031058/job/93946483060) |


## [Run #31541844088](https://github.com/sgl-project/sglang/actions/runs/31541844088)
- **分支**: `lsyin/dsv4-parser-fix`
- **总耗时**: 11.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31541844088

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979598) |
| base-b-test-2-npu-a3 / run (0) | 8.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979786) |
| base-b-test-16-npu-a3 / run (0) | 7.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979819) |
| base-b-test-4-npu-a3 / run (1) | 9.4min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979900) |
| base-b-test-1-npu-a3 / run (0) | 8.4min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979907) |
| base-b-test-4-npu-a3 / run (0) | 9.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979921) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945980267) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 10.3min | 环境问题 | 容器内安装依赖时失败，自托管runner执行自定义容器实现出错。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945980366) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945980368) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 3.8min | 环境问题 | 自定义容器执行失败，导致测试未开始即终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93947486971) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、Node版本警告及上传artifact时未找到diffusion-failures目录。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979598

- **base-b-test-2-npu-a3 / run (0)**: 日志显示容器在加载模型权重时出现"Executing the custom container implementation failed"错误，这是自托管runner环境问题，可能与容器配置或资源限制有关，而非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979786

- **base-b-test-16-npu-a3 / run (0)**: 日志显示服务启动正常，但随后报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于NPU容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979819

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型权重加载刚开始（0/16）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979900

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行到第3个文件时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于运行环境或容器配置问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979907

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系self hosted runner管理员，属于runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979921

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试请求均正常返回200，但随后出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945980267

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 在安装antlr4-python3-runtime构建依赖时，自定义容器执行失败，提示联系runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945980366

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945980368

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在运行测试前，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93947486971

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979857) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945979881) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31541844088/job/93945980238) |


## [Run #31541707720](https://github.com/sgl-project/sglang/actions/runs/31541707720)
- **分支**: `main`
- **总耗时**: 11.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31541707720

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482529) |
| base-b-test-16-npu-a3 / run (0) | 8.9min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482762) |
| base-b-test-2-npu-a3 / run (0) | 8.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482768) |
| base-b-test-1-npu-a3 / run (0) | 10.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482781) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482788) |
| base-b-test-4-npu-a3 / run (0) | 8.0min | 代码错误 | NPU HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482838) |
| base-b-test-4-npu-a3 / run (1) | 9.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482851) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945483249) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945483325) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.4min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945483338) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 4.3min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93946680141) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法看到具体测试命令和错误输出。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位真实原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482529

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动NPU容器后，TokenizerManager初始化过程中容器执行失败，报错'Executing the custom container implementation failed'，可能是NPU驱动或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482762

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU CI环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482768

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中（进度18%）时，runner报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482781

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试运行中容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482788

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 执行失败，0/5 测试通过，耗时281秒。可能是 HiCache MLA 功能实现存在 bug 或与当前 CANN 9.0.0 环境不兼容，需查看具体测试断言输出定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482838

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试在运行第二个测试文件时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482851

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945483249

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在模型加载和批处理捕获阶段，各TP/EP进程输出'Disable CP decode attention TP'后，容器执行失败，提示'Executing the custom container implementation failed'，属于NPU运行环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945483325

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945483338

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在启动NPU推理进程后，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93946680141

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945482837) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31541707720/job/93945483321) |


## [Run #31540857813](https://github.com/sgl-project/sglang/actions/runs/31540857813)
- **分支**: `main`
- **总耗时**: 10.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31540857813

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759531) |
| base-b-test-2-npu-a3 / run (0) | 9.1min | 环境问题 | 自定义容器启动失败，torch_npu 初始化异常导致作业中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759622) |
| base-b-test-4-npu-a3 / run (1) | 8.9min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759670) |
| base-b-test-8-npu-a3 / run (0) | 8.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759704) |
| base-b-test-1-npu-a3 / run (0) | 9.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759721) |
| base-b-test-16-npu-a3 / run (0) | 5.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759724) |
| base-b-test-4-npu-a3 / run (0) | 7.2min | 环境问题 | 测试进程触发CUDA coredump导致容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759806) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759886) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.9min | 环境问题 | 自定义容器执行失败，导致作业在安装依赖阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759932) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.2min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942760040) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（无文件上传），未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759531

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 torch_npu 的 transfer_to_npu 模块在导入时触发 ImportWarning 和 RuntimeWarning，随后自定义容器执行失败，提示联系自托管 runner 管理员，属于 NPU 环境配置或兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759622

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行到92%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759670

- **base-b-test-8-npu-a3 / run (0)**: 日志显示模型加载到62%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759704

- **base-b-test-1-npu-a3 / run (0)**: 日志显示torchair配置警告后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759721

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载到69%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于NPU CI环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759724

- **base-b-test-4-npu-a3 / run (0)**: 运行中Python进程发生崩溃（libc free/gc_collect相关），触发CUDA coredump机制，因未设置CUDA_ENABLE_USER_TRIGGERED_COREDUMP导致等待超时，最终容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759806

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但执行自定义容器时失败，错误为'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759886

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在pip安装evalscope依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759932

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942760040

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942759610) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31540857813/job/93942760049) |


## [Run #31539861635](https://github.com/sgl-project/sglang/actions/runs/31539861635)
- **分支**: `mmangkad/flashinfer-0.6.17rc1-kimi-k3`
- **总耗时**: 91.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31539861635

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93940564397) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93945765974) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93949500032) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93959087980) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1122秒后失败，0/1通过，属于性能基准测试未达标，可能因模型推理速度或延迟不满足50ms要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93940564397

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业触发了fast-fail，导致本作业未实际运行即被终止，属于环境或依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93945765974

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，本作业作为级联失败被跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93949500032

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业被级联跳过，未执行实际测试，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93959087980

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939662154) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939662389) |
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939662411) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939662495) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939662513) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939662519) |
| base-b-test-16-npu-a3 / run (0) | 46.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939662549) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939662609) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939663085) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939663145) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939663165) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31539861635/job/93939663203) |


## [Run #31538429514](https://github.com/sgl-project/sglang/actions/runs/31538429514)
- **分支**: `muse-glimmer`
- **总耗时**: 69.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31538429514

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 68.1min | 环境问题 | 自托管runner执行容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935160383) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93936236676) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 37.3min | 性能回归 | NPU性能测试中kimi_k2_6用例失败，4个测试仅1个通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93941380095) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935160383

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1087秒后失败，该测试为性能基准测试，预期耗时3600秒，但实际未通过，可能因性能未达阈值或运行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93936236676

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms测试未通过（exit code 1），耗时1574秒，可能因性能未达预期或运行错误导致，需检查具体日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93941380095

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935159644) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935159905) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935159912) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935159936) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935159989) |
| base-b-test-16-npu-a3 / run (0) | 52.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935160007) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935160010) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935160014) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935160349) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935160456) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93935160468) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31538429514/job/93944686093) |


## [Run #31535898510](https://github.com/sgl-project/sglang/actions/runs/31535898510)
- **分支**: `cherry-pick-zmq-max-sockets`
- **总耗时**: 84.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31535898510

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380430) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.3min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93929311644) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 34.3min | 性能回归 | NPU性能测试中kimi_k2_6用例失败，导致作业整体失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93933990799) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93937198683) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行约80分钟后，在Decode阶段正常输出时突然报错'Executing the custom container implementation failed'，属于runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380430

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1065秒后失败，该测试为性能基准测试，返回码1表明性能指标未达到预期要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93929311644

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件4个用例中1个失败，具体为kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms测试未通过（退出码1），耗时1455秒，可能因性能未达预期或运行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93933990799

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因作业，因此本作业（4-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93937198683

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927379880) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927379996) |
| base-b-test-16-npu-a3 / run (0) | 55.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380029) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380039) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380061) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380062) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380076) |
| base-b-test-1-npu-a3 / run (0) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380084) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380522) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380580) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31535898510/job/93927380621) |


## [Run #31534882954](https://github.com/sgl-project/sglang/actions/runs/31534882954)
- **分支**: `idhanani/dyn-29465-mm-inputs-msgpack`
- **总耗时**: 124.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31534882954

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.5min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93925310049) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 24.4min | 性能回归 | NPU性能测试未通过，qwen3_235b_w8a8_8p_in3k5_out1k5_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93929979231) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败机制跳过了本作业 | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93953205306) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1083秒后退出码1，0/1通过，属于性能指标未达标或执行错误，需检查模型配置或NPU环境。
  链接: https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93925310049

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中4个测试全部失败，其中qwen3_235b_w8a8_8p_in3k5_out1k5_50ms测试运行1241秒后退出码为1，未达到性能预期标准，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93929979231

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示该作业因其他两个作业（base-c-test-perf-8/16-npu-a3）失败而被健康检查快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93953205306

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600001) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600097) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600106) |
| base-b-test-2-npu-a3 / run (0) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600182) |
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600184) |
| base-b-test-16-npu-a3 / run (0) | 55.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600190) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600220) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600431) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600448) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600449) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 122.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923600450) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93923601057) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31534882954/job/93934053907) |


## [Run #31534412786](https://github.com/sgl-project/sglang/actions/runs/31534412786)
- **分支**: `feat/add-cosmos3-edge-distil`
- **总耗时**: 25.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31534412786

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31534412786/job/93922111964) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31534412786/job/93922111964


## [Run #31530803997](https://github.com/sgl-project/sglang/actions/runs/31530803997)
- **分支**: `kda-helion-backend`
- **总耗时**: 126.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31530803997

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 35.2min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，未达性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93921174410) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败导致快速失败，本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93946471052) |

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件4个用例中1个失败，qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例退出码1，耗时1274秒，未通过性能基准，其余3个用例通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93921174410

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-16-npu-a3 作业失败，触发 fast-fail 机制，本作业未实际运行即被终止，属于上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93946471052

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914296562) |
| base-b-test-1-npu-a3 / run (0) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914296707) |
| base-b-test-4-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914296796) |
| base-b-test-4-npu-a3 / run (1) | 13.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914296835) |
| base-b-test-16-npu-a3 / run (0) | 51.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914296847) |
| base-b-test-2-npu-a3 / run (0) | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914296851) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914296898) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914297066) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914297067) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914297097) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 122.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914297143) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93914297196) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93915365969) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31530803997/job/93925740582) |


## [Run #31529132656](https://github.com/sgl-project/sglang/actions/runs/31529132656)
- **分支**: `cheng/gc-sr-review`
- **总耗时**: 128.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31529132656

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.0min | 其他 | 作业日志不完整，仅显示上传artifact步骤，未包含实际测试执行内容。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774566) |
| base-b-test-16-npu-a3 / run (0) | 65.9min | 代码错误 | NPU PD分离测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774602) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 127.9min | 精度回归 | qwen3_5_9b 测试失败，精度未达标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904775191) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | 性能回归 | NPU性能测试未达标，minimax_m2_5测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93905917850) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 33.3min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，未达性能指标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93911269786) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions基础设施信息（Node版本警告、artifact上传等），未显示multimodal-gen-test-2-npu-a3作业的实际测试命令、输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774566

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（exit code 1），耗时303秒，其余3个测试均通过。可能是PD分离功能相关代码或测试用例本身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774602

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在 NPU A3 精度测试中，qwen3_5_9b_bf16_1p_gsm8k 用例退出码为1，而其他两个用例通过，表明该模型存在精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904775191

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1144秒后失败，0/1通过，属于性能测试未达到预期标准。
  链接: https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93905917850

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件4个用例中1个失败，qwen3_235b_w8a8_8p_in3k5_out1k5_50ms测试未通过，耗时1227秒，可能因性能未达50ms目标或运行错误导致退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93911269786

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774442) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774670) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774692) |
| base-b-test-2-npu-a3 / run (0) | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774710) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774792) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774860) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904774911) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904775131) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904775236) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93904775271) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31529132656/job/93915096164) |


## [Run #31526830059](https://github.com/sgl-project/sglang/actions/runs/31526830059)
- **分支**: `shreyasm/fix-dsa-prefill-cp-num-splits`
- **总耗时**: 86.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31526830059

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试用例失败，未达到预期性能指标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93909442499) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93915831546) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93918610630) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93930955928) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，运行1122秒后退出码为1，属于性能测试未通过，可能因模型推理性能未达标或结果校验失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93909442499

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因失败，因此本作业（base-c-test-perf-16-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93915831546

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，实际未执行测试，属于上游失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93918610630

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93930955928

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907782278) |
| base-b-test-16-npu-a3 / run (0) | 47.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907782522) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907782544) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907782636) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907782705) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907782729) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907782782) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907782849) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907783333) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907783413) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907783534) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31526830059/job/93907783611) |


---
*Auto-generated by npu_pr_monitor.py*