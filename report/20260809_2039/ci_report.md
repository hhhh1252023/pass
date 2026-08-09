# NPU CI 执行监控
**生成时间**: 2026-08-09 12:39 UTC
**分析 Run 数**: 28

---

## [Run #31310133750](https://github.com/sgl-project/sglang/actions/runs/31310133750)
- **分支**: `dsv4-topk-cuda13-dsmem-fix`
- **总耗时**: 11.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31310133750

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 10.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606954) |
| base-b-test-16-npu-a3 / run (0) | 8.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606960) |
| base-b-test-8-npu-a3 / run (0) | 6.0min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606966) |
| base-b-test-4-npu-a3 / run (1) | 7.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606970) |
| base-b-test-2-npu-a3 / run (0) | 7.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606983) |
| base-b-test-4-npu-a3 / run (0) | 6.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606996) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236607162) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.8min | 环境问题 | 自定义容器执行失败，模型权重加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236607163) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236607171) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.7min | 其他 | 健康检查中的lint检查失败导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93237300761) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示torch_npu的transfer_to_npu模块在容器启动时产生ImportWarning和RuntimeWarning，随后自定义容器实现执行失败，导致作业无法正常运行，属于NPU容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606954

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606960

- **base-b-test-8-npu-a3 / run (0)**: 作业在加载权重阶段（TP0-EP0等）后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606966

- **base-b-test-4-npu-a3 / run (1)**: 日志显示torch_npu初始化时出现ImportWarning和RuntimeWarning，随后容器执行失败，错误为'Executing the custom container implementation failed'，属于NPU容器环境配置或兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606970

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重后，执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606983

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试本身通过（Ran 1 test OK），但在运行第二个测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606996

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在加载模型分片过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236607162

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载模型分片（约62%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境或资源问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236607163

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型权重加载完成后，在NPU环境初始化阶段（获取ASCEND_OPP_PATH等）出现错误，导致自定义容器实现执行失败，属于NPU环境配置或容器兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236607171

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业立即退出，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93237300761

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236606965) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31310133750/job/93236607147) |


## [Run #31307671542](https://github.com/sgl-project/sglang/actions/runs/31307671542)
- **分支**: `main`
- **总耗时**: 86.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31307671542

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 9.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536593) |
| base-b-test-16-npu-a3 / run (0) | 42.8min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536620) |
| base-b-test-1-npu-a3 / run (0) | 5.8min | 精度回归 | HiCache MHA测试精度失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536625) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 精度回归 | HiCache MLA测试精度失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536644) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536761) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230911178) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93234780440) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93239001961) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536593

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时353秒，可能因代码逻辑或环境问题导致，需查看具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536620

- **base-b-test-1-npu-a3 / run (0)**: test_npu_hicache_mha.py测试返回exit code 1，测试摘要显示0/11通过，该测试涉及Qwen2.5-7B-Instruct模型精度验证，可能因模型输出与预期不符导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536625

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试在DeepSeek-V2-Lite-W8A8模型上精度校验失败（errors=1），导致0/5测试通过，作业退出码255。
  链接: https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536644

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在“Check PR test health”步骤失败，原因是multimodal-gen-test-2-npu-a3、base-b-test-1-npu-a3和base-b-test-4-npu-a3等根因作业失败，触发了快速失败机制，本作业并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536761

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现multimodal-gen-test-2-npu-a3、base-b-test-1-npu-a3等根因作业失败，触发fast-fail机制，本作业未实际运行即被取消，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230911178

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在“Check PR test health”步骤因检测到其他根因作业（如multimodal-gen-test-2-npu-a3等）失败而触发fast-fail，本作业未实际运行测试，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93234780440

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在启动前执行PR测试健康检查，检测到多个根因作业（如multimodal-gen-test-2-npu-a3等）失败，触发快速失败机制，本作业未实际运行测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93239001961

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536563) |
| base-b-test-2-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536617) |
| base-b-test-8-npu-a3 / run (0) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536629) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536635) |
| base-b-test-4-npu-a3 / run (1) | 28.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536720) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 80.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536754) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536762) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31307671542/job/93230536786) |


## [Run #31306800831](https://github.com/sgl-project/sglang/actions/runs/31306800831)
- **分支**: `kp/fix-triton-dcp-gqa-prefill`
- **总耗时**: 63.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31306800831

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 22.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398932) |
| base-b-test-16-npu-a3 / run (0) | 32.2min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398959) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.9min | 精度回归 | Qwen3.5-9B GSM8K 测试精度低于基线，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228399040) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93229448693) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93231533295) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93233072604) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398932

- **base-b-test-16-npu-a3 / run (0)**: 测试套件中3/6通过，但test_npu_pd_disaggregation.py返回退出码1，耗时376秒，未显示具体错误信息，需进一步查看该测试日志定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398959

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 TestNPUQwen3_5_9B_GSM8K 的 accuracy 为 0.81，低于基线 0.835，未达到精度要求，测试用例返回退出码 1，最终作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228399040

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量382.75，低于基线390.5859，未通过性能阈值，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93229448693

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到multimodal-gen-test-2-npu-a3、base-c-test-acc-2-npu-a3、base-c-test-perf-8-npu-a3等根因作业失败，触发fast-fail机制，本作业未实际执行即被跳过，属于依赖的上游失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93231533295

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: PR健康检查检测到multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3等4个根因作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93233072604

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398929) |
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398954) |
| base-b-test-4-npu-a3 / run (0) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398962) |
| base-b-test-4-npu-a3 / run (1) | 12.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398964) |
| base-b-test-2-npu-a3 / run (0) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398971) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398983) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228398990) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228399061) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228399062) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306800831/job/93228399068) |


## [Run #31306796375](https://github.com/sgl-project/sglang/actions/runs/31306796375)
- **分支**: `refactor/unwrap-staging-register-info`
- **总耗时**: 34.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31306796375

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399084) |
| multimodal-gen-test-1-npu-a3 | 26.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399088) |
| base-b-test-16-npu-a3 / run (0) | 26.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399137) |
| base-b-test-4-npu-a3 / run (0) | 23.1min | 环境问题 | 自定义容器执行失败，NPU分布式初始化未完成 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399161) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399198) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 25.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399209) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93229640184) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93231472425) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤（无文件上传），未出现任何测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399084

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件（No files were found），说明测试可能未产生失败样本或提前退出，但根因无法从当前日志判断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399088

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型分片到61%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399137

- **base-b-test-4-npu-a3 / run (0)**: 日志显示torch分布式初始化刚开始（TP1 Init torch distributed begin）即报错，提示自定义容器实现执行失败，可能是NPU环境或容器配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399161

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在运行约25分钟后，日志显示"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399198

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399209

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在性能测试运行过程中，GitHub Actions 报错“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于容器环境问题，而非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93229640184

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，属于基础设施/环境配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93231472425

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399130) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399132) |
| base-b-test-2-npu-a3 / run (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399138) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399150) |
| base-b-test-1-npu-a3 / run (0) | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399166) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399211) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31306796375/job/93228399231) |


## [Run #31304740337](https://github.com/sgl-project/sglang/actions/runs/31304740337)
- **分支**: `revert-pr-32588-lifecycle`
- **总耗时**: 73.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31304740337

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 24.6min | 其他 | 作业未显示明确失败原因，仅上传artifact时未找到文件，可能测试未执行或结果为空。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226164) |
| base-b-test-16-npu-a3 / run (0) | 33.5min | 代码错误 | NPU PD分离测试失败，3/6用例通过，1个测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226184) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 70.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226286) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | 性能回归 | 性能测试未通过，吞吐量未达到基线要求。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223948206) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 23.9min | 性能回归 | 性能测试未达到基线，测试用例失败导致作业退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93225843232) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93227472278) |

- **multimodal-gen-test-2-npu-a3**: 日志显示upload-artifact步骤提示未找到diffusion-failures/目录，说明测试可能未产生失败样本或测试未运行。作业整体未出现明显错误或超时，需进一步查看测试执行日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226164

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时345秒，退出码1。其他3个测试通过，非环境或超时问题，属于该测试用例的代码或功能错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226184

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在10:10:14时出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226286

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms失败，实际吞吐量392.61，基线为390.5859，虽略高于基线但测试仍返回退出码1，可能因其他性能指标或稳定性要求未满足。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223948206

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试用例test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py返回退出码1，实际吞吐量6214.82低于基线6189.0，未通过性能基准，导致0/4测试通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93225843232

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业未实际运行，而是因健康检查检测到其他根因作业（如multimodal-gen-test-2-npu-a3等）失败，触发了快速失败机制，导致本作业被跳过并报错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93227472278

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226157) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226177) |
| base-b-test-2-npu-a3 / run (0) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226181) |
| base-b-test-4-npu-a3 / run (0) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226188) |
| base-b-test-8-npu-a3 / run (0) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226196) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226205) |
| base-a-test-1-npu-a2 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226206) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226275) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226279) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304740337/job/93223226284) |


## [Run #31304382522](https://github.com/sgl-project/sglang/actions/runs/31304382522)
- **分支**: `patch-4`
- **总耗时**: 105.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31304382522

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316228) |
| base-b-test-16-npu-a3 / run (0) | 35.4min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316319) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.3min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，其余两项通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316486) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.8min | 性能回归 | 性能测试未达到基线，吞吐量低于预期导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93223627930) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 36.6min | 性能回归 | 性能测试用例 kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms 失败，未达到预期性能指标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93225351590) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行步骤或失败断言，仅显示runner启动、Node版本警告及上传artifact时无文件。可能因日志截断或作业在测试前被取消，需查看完整日志定位真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316228

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，耗时333秒，其余3个测试通过，表明该测试用例存在代码或逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316319

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码1，耗时1445秒，未达预期精度，疑似模型精度回归或数据/配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316486

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量389.05，低于基线390.5859，未通过性能阈值检查，测试返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93223627930

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中 4 个用例仅 1 个通过，kimi_k2_6 用例返回退出码 1，耗时 1504 秒，未满足 20ms 延迟目标，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93225351590

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316244) |
| base-b-test-4-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316298) |
| base-b-test-2-npu-a3 / run (0) | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316304) |
| base-b-test-4-npu-a3 / run (1) | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316306) |
| base-b-test-1-npu-a3 / run (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316350) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316352) |
| base-a-test-1-npu-a2 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316361) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316438) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316439) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93222316455) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31304382522/job/93227897874) |


## [Run #31303516543](https://github.com/sgl-project/sglang/actions/runs/31303516543)
- **分支**: `main`
- **总耗时**: 21.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31303516543

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 9.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303516543/job/93220105576) |
| multimodal-gen-test-1-npu-a3 | 10.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303516543/job/93220105618) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303516543/job/93220105576

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（无文件上传），未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303516543/job/93220105618


## [Run #31303467897](https://github.com/sgl-project/sglang/actions/runs/31303467897)
- **分支**: `kda-cp-state-preprocess`
- **总耗时**: 91.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31303467897

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 23.9min | 其他 | 作业未执行实际测试，仅上传空失败目录后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954442) |
| base-b-test-16-npu-a3 / run (0) | 36.0min | 代码错误 | NPU PD分离测试失败，3/6用例通过，1个测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954483) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.8min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，2/3 通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954671) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.7min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93220368574) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.4min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93222797678) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 作业因其他根因作业失败被快速失败机制跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93225246882) |

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未发现测试执行记录，且diffusion-failures目录无文件，最终正常清理退出，无明确失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954442

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py 测试失败（耗时376秒），其余3个测试通过。可能是该测试用例存在逻辑错误或环境配置问题，需查看具体断言失败信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954483

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码 1，导致整体作业失败。其他两个测试通过，表明是特定模型精度未达标，而非环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954671

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量377.35，低于基线390.5859，性能回归约3.4%，未通过性能阈值检查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93220368574

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现其他根因作业（如multimodal-gen-test-2-npu-a3等）失败，触发fast-fail机制，本作业未实际运行即被跳过，属于依赖的上游失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93222797678

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业本身未执行，因健康检查发现其他作业（multimodal-gen-test-2-npu-a3等）失败，触发fast-fail机制，导致本作业被跳过并报错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93225246882

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954441) |
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954484) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954487) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954498) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954510) |
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954514) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954539) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954610) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 43.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954680) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303467897/job/93219954686) |


## [Run #31303402930](https://github.com/sgl-project/sglang/actions/runs/31303402930)
- **分支**: `elastic-ep-cuda-graph-recapture`
- **总耗时**: 85.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31303402930

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 26.4min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783227) |
| base-b-test-16-npu-a3 / run (0) | 35.6min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783322) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.2min | 精度回归 | NPU精度测试中qwen3_5_9b用例失败，其余两个用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783467) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.2min | 性能回归 | 性能测试未通过，吞吐量未达到基线要求 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93220308410) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.2min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93222393090) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93224529079) |

- **multimodal-gen-test-2-npu-a3**: 日志中只有GitHub Actions的初始化、Node版本警告和上传artifact步骤，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783227

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，3/6测试通过，该用例耗时359秒后失败，可能涉及PD分离功能逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783322

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件base-c-test-acc-2-npu-a3中，qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，导致整体作业失败。另外两个用例（glm4_7_flash和moonshotai_moonlight_16b）均通过，表明是特定模型精度问题，而非环境或基础设施故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783467

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量396.57，基线为390.5859，但测试仍失败，可能因其他性能指标（如延迟）未达标或测试脚本判定逻辑问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93220308410

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业未实际运行，而是因健康检查检测到其他根因作业（如multimodal-gen-test-2-npu-a3等）失败，触发了快速失败机制，导致本作业被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93222393090

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在健康检查阶段检测到其他三个作业（multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3、base-c-test-perf-8-npu-a3）失败，触发Fast-fail机制，导致本作业被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93224529079

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783223) |
| base-b-test-4-npu-a3 / run (1) | 12.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783254) |
| base-b-test-4-npu-a3 / run (0) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783257) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783288) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783311) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783324) |
| base-b-test-2-npu-a3 / run (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783325) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783437) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783445) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303402930/job/93219783448) |


## [Run #31303204831](https://github.com/sgl-project/sglang/actions/runs/31303204831)
- **分支**: `main`
- **总耗时**: 8.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31303204831

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.5min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219363963) |
| multimodal-gen-test-2-npu-a3 | 6.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219363978) |
| base-b-test-8-npu-a3 / run (0) | 6.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364020) |
| base-b-test-1-npu-a3 / run (0) | 6.1min | 代码错误 | NPU HiCache MHA 测试失败，0/11 用例通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364025) |
| base-b-test-16-npu-a3 / run (0) | 5.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364043) |
| base-b-test-4-npu-a3 / run (1) | 5.3min | 环境问题 | 自定义容器执行失败，模型权重加载中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364049) |
| base-b-test-2-npu-a3 / run (0) | 5.3min | 环境问题 | 自定义容器执行失败，NPU作业在模型加载阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364058) |
| base-b-test-4-npu-a3 / run (0) | 4.0min | 环境问题 | 自定义容器启动失败，导致作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364066) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364153) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364169) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.9min | 环境问题 | 自定义容器执行失败，导致测试未开始即终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364201) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 2.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219790299) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219363963

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219363978

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试正常运行中，但突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题而非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364020

- **base-b-test-1-npu-a3 / run (0)**: 测试 test_npu_hicache_mha.py 返回退出码 1，所有 11 个测试均未通过，可能是 HiCache 功能在 NPU 上存在实现问题或配置错误，需检查相关代码和日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364025

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型分片时（38%）容器执行失败，报错提示联系自托管runner管理员，属于NPU测试环境基础设施问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364043

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在加载模型权重分片时（14%进度）出现错误，提示自定义容器实现失败，可能是NPU环境或容器配置问题导致作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364049

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型分片加载至81%时，runner报错“Executing the custom container implementation failed”，随后进入清理流程。这属于自托管runner容器环境异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364058

- **base-b-test-4-npu-a3 / run (0)**: 日志显示执行自定义容器实现时失败，提示联系自托管runner管理员，属于NPU CI环境配置或容器镜像问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364066

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型权重加载到35%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364153

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364169

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动测试时，执行自定义容器实现失败，错误信息为“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364201

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是构建产物未上传或存储配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219790299

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364024) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31303204831/job/93219364204) |


## [Run #31302999540](https://github.com/sgl-project/sglang/actions/runs/31302999540)
- **分支**: `main`
- **总耗时**: 5.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31302999540

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 0.6min | 环境问题 | 自托管runner的k8s容器启动失败，导致作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770886) |
| base-b-test-4-npu-a3 / run (1) | 4.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770909) |
| base-b-test-4-npu-a3 / run (0) | 4.2min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770921) |
| base-b-test-8-npu-a3 / run (0) | 4.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770927) |
| base-b-test-1-npu-a3 / run (0) | 3.9min | 环境问题 | 自定义容器启动失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770938) |
| base-b-test-16-npu-a3 / run (0) | 3.3min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770948) |
| base-b-test-2-npu-a3 / run (0) | 4.0min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770976) |
| base-a-test-1-npu-a2 / run (0) | 4.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770977) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.1min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218771034) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.4min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218771035) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.0min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218771054) |

- **multimodal-gen-test-1-npu-a3**: 日志显示prepareJob阶段执行自定义容器实现失败，错误为'jobPod must be set'，说明k8s pod未成功创建，属于runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770886

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动NPU容器时失败，日志显示模型配置警告后容器执行报错，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770909

- **base-b-test-4-npu-a3 / run (0)**: 测试刚开始执行（test_a_gsm8k）时，自定义容器实现失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770921

- **base-b-test-8-npu-a3 / run (0)**: 日志显示在TokenizerManager初始化后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770927

- **base-b-test-1-npu-a3 / run (0)**: 日志显示容器初始化过程中出现错误，最终报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于NPU环境配置或容器镜像问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770938

- **base-b-test-16-npu-a3 / run (0)**: 作业在运行测试前执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770948

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在启动自定义容器时出现错误："Executing the custom container implementation failed"，这是自托管runner环境问题，可能是容器镜像拉取或配置错误，并非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770976

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218770977

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在启动测试容器时出现错误："Executing the custom container implementation failed"，随后作业清理退出。这属于自托管runner环境或容器配置问题，而非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218771034

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在测试启动后不久，出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境或runner配置问题，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218771035

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218771054

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31302999540/job/93218771021) |


## [Run #31301846518](https://github.com/sgl-project/sglang/actions/runs/31301846518)
- **分支**: `revert-pr-32588-lifecycle`
- **总耗时**: 68.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31301846518

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 9.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255529) |
| base-b-test-16-npu-a3 / run (0) | 36.3min | 代码错误 | NPU PD disaggregation 测试失败，3/6 通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255566) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 67.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255626) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.4min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216736776) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93219308253) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93220795144) |

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传artifact时未找到diffusion-failures目录，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255529

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py 测试返回退出码 1，耗时 358 秒，其余 3 个测试通过。该测试涉及 PD 分离功能，可能因代码逻辑或环境配置问题导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255566

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255626

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量390.25，低于基线390.5859，未通过性能阈值，测试返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216736776

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 本作业未实际运行，被健康检查脚本因检测到其他根因作业（multimodal-gen-test-2-npu-a3和base-c-test-perf-8-npu-a3）失败而快速跳过，属于级联取消，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93219308253

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在健康检查阶段检测到其他三个作业（multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3、base-c-test-perf-8-npu-a3）失败，触发Fast-fail机制，导致本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93220795144

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255525) |
| base-b-test-2-npu-a3 / run (0) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255546) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255554) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255559) |
| base-b-test-4-npu-a3 / run (0) | 31.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255561) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255567) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255572) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255649) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255667) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301846518/job/93216255671) |


## [Run #31301493136](https://github.com/sgl-project/sglang/actions/runs/31301493136)
- **分支**: `hisparse_mtp_kernel`
- **总耗时**: 96.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31301493136

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 35.5min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214921958) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.7min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93215326233) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 健康检查快速失败，因另一作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93217174491) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93219880649) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93224428729) |

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，3/6测试通过，该用例耗时330秒失败，其余用例均通过，属于该测试用例本身的代码或逻辑问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214921958

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实测吞吐量375.05，低于基线390.5859，性能回归约4%，导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93215326233

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 本作业未实际运行，因健康检查检测到同PR中base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，跳过本作业并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93217174491

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在“Check PR test health”步骤失败，原因是根因作业base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3失败，触发了快速失败机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93219880649

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业未实际运行，因健康检查检测到根因作业（base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3）失败，触发快速失败机制，本作业被跳过，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93224428729

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214921945) |
| base-b-test-4-npu-a3 / run (0) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214921951) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214921954) |
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214921960) |
| base-b-test-2-npu-a3 / run (0) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214921961) |
| base-a-test-1-npu-a2 / run (0) | 10.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214921963) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214922060) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214922103) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214922107) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301493136/job/93214922139) |


## [Run #31301026210](https://github.com/sgl-project/sglang/actions/runs/31301026210)
- **分支**: `feature/load-reporter`
- **总耗时**: 93.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31301026210

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 23.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742441) |
| base-b-test-16-npu-a3 / run (0) | 32.9min | 代码错误 | NPU PD分离测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742597) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.2min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，2/3 通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742662) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.2min | 精度回归 | GLM5 GSM8K 测试精度低于基线，导致测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742691) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | 性能测试未达到基线，测试用例失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93214211576) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93217824269) |

- **multimodal-gen-test-2-npu-a3**: 日志中仅包含Node.js版本弃用警告、上传diffusion-failures目录时无文件等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因，可能为日志截断或作业被外部取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742441

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，3/6测试通过，该测试用例本身存在代码或逻辑错误导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742597

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码 1，耗时 1240 秒，未达预期精度标准，其余两个测试通过，判定为精度回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742662

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: TestNPUGLM5_Top64_Pruned_GSM8K 测试精度为0.47，低于基线0.48，未达到精度要求，测试脚本返回退出码1，导致整个CI作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742691

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms的吞吐量为392.66，低于基线390.5859，导致测试失败，退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93214211576

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业未实际运行，因健康检查检测到multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3等4个根因作业失败，触发Fast-fail跳过本作业，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93217824269

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742433) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742511) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742522) |
| base-b-test-2-npu-a3 / run (0) | 19.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742541) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742549) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742564) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742578) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742682) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301026210/job/93213742699) |


## [Run #31301016716](https://github.com/sgl-project/sglang/actions/runs/31301016716)
- **分支**: `main`
- **总耗时**: 47.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31301016716

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.6min | 其他 | 作业未执行实际测试，仅上传空失败产物后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713908) |
| base-b-test-1-npu-a3 / run (0) | 6.0min | 代码错误 | NPU HiCache MHA 测试失败，测试用例报错退出 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713935) |
| base-b-test-16-npu-a3 / run (0) | 46.3min | 代码错误 | NPU PD分离测试用例失败，导致作业整体退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713938) |
| base-b-test-4-npu-a3 / run (0) | 9.1min | 代码错误 | HiCache MLA测试失败，服务启动后测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713963) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.8min | 精度回归 | Qwen3.5-9B GSM8K 测试精度不达标，准确率0.77低于基线0.835。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213714091) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93214120068) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93216415631) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93217878721) |

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，随后正常清理退出。未出现测试运行、错误或超时信息，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713908

- **base-b-test-1-npu-a3 / run (0)**: test_npu_hicache_mha.py 测试在运行146秒后失败，返回错误码1，导致整个测试套件0/11通过。可能是测试代码或环境配置问题，需查看具体错误详情。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713935

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py返回退出码1，耗时341秒，其余3个测试通过。该测试可能因代码逻辑或环境配置问题失败，需查看具体错误日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713938

- **base-b-test-4-npu-a3 / run (0)**: 测试test_npu_hicache_mla.py在启动DeepSeek-V2-Lite-W8A8模型服务后执行失败，退出码1，导致整体测试0/5通过。可能是HiCache功能或MLA相关代码存在bug，需检查测试日志定位具体断言失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713963

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试用例 test_npu_qwen3_5_9b_bf16_1p_gsm8k.py 失败，实际准确率0.77低于基线0.835，导致测试未通过，作业以退出码1结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213714091

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示health-check检测到base-b-test-4-npu-a3作业失败，将其视为根因作业，触发了fast-fail机制，导致本作业在启动前被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93214120068

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业因PR健康检查检测到其他根因作业（multimodal-gen-test-2-npu-a3等）失败而触发快速失败机制，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93216415631

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 本作业在“Check PR test health”步骤中检测到其他根因作业（如multimodal-gen-test-2-npu-a3等）失败，触发了快速失败机制，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93217878721

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 35.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713890) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713924) |
| base-b-test-8-npu-a3 / run (0) | 11.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713926) |
| base-b-test-2-npu-a3 / run (0) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713943) |
| base-b-test-4-npu-a3 / run (1) | 27.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213713956) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213714082) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213714098) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31301016716/job/93213714099) |


## [Run #31300214702](https://github.com/sgl-project/sglang/actions/runs/31300214702)
- **分支**: `cheng/gc-rc-review`
- **总耗时**: 82.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31300214702

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 22.6min | 其他 | 作业日志不完整，未显示测试执行过程，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770475) |
| base-b-test-16-npu-a3 / run (0) | 36.6min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试用例返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770583) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 72.9min | 精度回归 | NPU精度测试中qwen3_5_9b_bf16_1p_gsm8k用例失败，导致作业整体失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770704) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.2min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93212381463) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 43.2min | 性能回归 | NPU性能测试中qwen3_235b_a22b用例失败，导致整体作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93214388987) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含实际测试命令或失败断言，仅显示上传diffusion-failures目录时无文件，可能测试未运行或日志被截断，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770475

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时382秒，退出码1，其余3个测试通过。可能是PD分离功能相关代码或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770583

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中3个用例仅1个通过，qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，耗时1383秒，未达预期精度标准，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770704

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量389.35，基线为390.5859，略低于基线，判定为性能回归，测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93212381463

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中2/4通过，qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例返回退出码1，耗时1290秒，未达预期性能标准，疑似性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93214388987

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770465) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770503) |
| base-b-test-4-npu-a3 / run (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770517) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770557) |
| base-b-test-4-npu-a3 / run (1) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770560) |
| base-b-test-2-npu-a3 / run (0) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770596) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770597) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770722) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770729) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93211770741) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31300214702/job/93215641658) |


## [Run #31299992118](https://github.com/sgl-project/sglang/actions/runs/31299992118)
- **分支**: `fix_glm52_pp`
- **总耗时**: 71.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31299992118

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 17.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141308) |
| base-b-test-16-npu-a3 / run (0) | 39.8min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试用例返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141345) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 70.5min | 精度回归 | NPU精度测试中qwen3_5_9b_bf16_1p_gsm8k用例失败，导致作业整体失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141477) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.0min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211624107) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败机制触发，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93213774581) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.1min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93215654968) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前结束，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141308

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时369秒，退出码1，其余3个测试通过，可能涉及PD分离功能逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141345

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中1/3用例通过，qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，耗时1326秒，未达预期精度标准，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141477

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实测吞吐量370.91，低于基线390.5859，性能回归约5%，导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211624107

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现multimodal-gen-test-2-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，触发了fast-fail机制，本作业被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93213774581

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业：multimodal-gen-test-2-npu-a3、base-b-test-16-npu-a3/run(0)、base-c-test-perf-8-npu-a3，本作业作为级联失败被跳过，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93215654968

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141286) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141323) |
| base-b-test-2-npu-a3 / run (0) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141352) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141354) |
| base-b-test-4-npu-a3 / run (0) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141372) |
| base-a-test-1-npu-a2 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141383) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141394) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 43.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141452) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141476) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299992118/job/93211141501) |


## [Run #31299216580](https://github.com/sgl-project/sglang/actions/runs/31299216580)
- **分支**: `cheng/gc-rc-review`
- **总耗时**: 27.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31299216580

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 12.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247559) |
| multimodal-gen-test-1-npu-a3 | 21.2min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247561) |
| base-b-test-2-npu-a3 / run (0) | 21.6min | 其他 | 作业实际成功，所有测试通过，仅出现Node.js 20弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247564) |
| base-b-test-16-npu-a3 / run (0) | 20.0min | 超时 | 模型权重加载时发生崩溃，导致Scheduler watchdog超时，作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247602) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247614) |
| base-b-test-4-npu-a3 / run (0) | 23.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247657) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 23.6min | 环境问题 | 自定义容器执行失败，测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247713) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 21.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247738) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 17.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247748) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 19.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209884602) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247559

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅有Node.js 20弃用警告和diffusion-failures目录无文件上传提示。作业可能因测试通过但无失败产物而正常结束，或失败原因未在日志中体现。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247561

- **base-b-test-2-npu-a3 / run (0)**: 日志显示6个NPU测试全部通过（passed: true），作业正常完成。仅有Node.js 20弃用警告和Buffer弃用警告，不影响结果，可能为误报或基础设施警告。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247564

- **base-b-test-16-npu-a3 / run (0)**: 在加载MoE模型权重时，copy_操作触发libtorch崩溃，随后Scheduler watchdog超时（300秒），最终自定义容器执行失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247602

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重后，执行自定义容器实现时失败，报错提示联系自托管runner管理员，可能是NPU设备或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247614

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在模型加载过程中（加载到55%时），GitHub Actions 报错“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于运行环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247657

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（解码吞吐约450 token/s），但在07:01:57时容器执行失败，错误为"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247713

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247738

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247748

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试运行中突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209884602

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247583) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247597) |
| base-b-test-4-npu-a3 / run (1) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247637) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299216580/job/93209247722) |


## [Run #31299009443](https://github.com/sgl-project/sglang/actions/runs/31299009443)
- **分支**: `cleanup/logits-processor-helpers`
- **总耗时**: 53.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31299009443

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 22.2min | 其他 | 作业未执行实际测试，仅上传空失败产物后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694334) |
| base-b-test-16-npu-a3 / run (0) | 35.0min | 代码错误 | NPU PD分离测试失败，3/6通过，1个测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694428) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 52.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694560) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.0min | 性能回归 | NPU性能测试未达基线，测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93209077547) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 26.7min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93210910646) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 5.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93213204654) |

- **multimodal-gen-test-2-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，随后正常清理退出。未出现测试运行、错误或超时信息，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694334

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时324秒，未通过。其他3个测试通过，失败原因可能涉及PD分离功能逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694428

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在07:23:07出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694560

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量394.2低于基线390.5859，未通过性能阈值检查，导致整个作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93209077547

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试用例TestQwen235B实际吞吐量5869.02，低于基线6189.0，未通过性能阈值检查，导致测试脚本退出码1，整体作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93210910646

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到，可能是文件被删除、路径错误或上传失败，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93213204654

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694305) |
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694367) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694386) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694388) |
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694399) |
| base-b-test-4-npu-a3 / run (0) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694412) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694438) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694574) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694587) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31299009443/job/93208694595) |


## [Run #31298802476](https://github.com/sgl-project/sglang/actions/runs/31298802476)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 92.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31298802476

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 22.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174911) |
| base-b-test-16-npu-a3 / run (0) | 35.0min | 代码错误 | NPU PD分离测试用例失败，3/6通过，1个测试文件返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174934) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.1min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，2/3通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208175124) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208580919) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93210502048) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93212265928) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤（未找到文件），未展示multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174911

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时341秒，具体错误信息未在日志中显示，但该测试文件返回退出码1，导致整个作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174934

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码1，耗时1275秒，未达预期精度标准，其余两个测试通过，判定为精度回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208175124

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量397.47，低于基线390.5859，未通过性能阈值检查，脚本退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208580919

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动前的健康检查中检测到同一PR的另一个作业multimodal-gen-test-2-npu-a3失败，根据快速失败策略，本作业被跳过并报错，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93210502048

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在健康检查阶段因检测到其他根因作业（如multimodal-gen-test-2-npu-a3等）失败而被快速失败机制跳过，自身未执行实际测试，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93212265928

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174892) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174939) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174944) |
| base-b-test-4-npu-a3 / run (0) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174948) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174976) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174981) |
| base-b-test-1-npu-a3 / run (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208174982) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208175119) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208175158) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298802476/job/93208175167) |


## [Run #31298521662](https://github.com/sgl-project/sglang/actions/runs/31298521662)
- **分支**: `cheng/gc-rc-review`
- **总耗时**: 18.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31298521662

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511923) |
| multimodal-gen-test-2-npu-a3 | 17.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511927) |
| base-b-test-2-npu-a3 / run (0) | 17.1min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511934) |
| base-b-test-16-npu-a3 / run (0) | 10.3min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511938) |
| base-b-test-1-npu-a3 / run (0) | 17.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511955) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511972) |
| base-b-test-4-npu-a3 / run (0) | 12.3min | 环境问题 | NPU容器启动后健康检查失败，自定义容器执行报错。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511983) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207512039) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207512046) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.9min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207512055) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93208619366) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions的初始化、依赖下载和上传artifact步骤，未包含multimodal-gen-test的实际执行输出。上传diffusion-failures目录时提示无文件，说明测试可能未运行或未产生失败文件，但无法从日志确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511923

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js 20弃用警告和上传失败文件未找到的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511927

- **base-b-test-2-npu-a3 / run (0)**: 测试运行到第3个文件时，自定义容器实现执行失败，报错'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511934

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是NPU环境或容器配置问题，而非代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511938

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在加载模型权重时出现"Executing the custom container implementation failed"错误，提示联系self hosted runner管理员，属于NPU容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511955

- **base-b-test-4-npu-a3 / run (1)**: 作业在模型权重加载阶段（27%）时，自定义容器实现执行失败，导致测试提前终止。日志显示为self-hosted runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511972

- **base-b-test-4-npu-a3 / run (0)**: 日志显示服务启动后/health_generate返回503，随后自定义容器实现执行失败，可能是NPU设备或容器环境配置问题导致服务未就绪。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511983

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在请求处理过程中出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207512039

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207512046

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常（吞吐约195 token/s），但突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207512055

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93208619366

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511963) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207511971) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31298521662/job/93207512045) |


## [Run #31297749345](https://github.com/sgl-project/sglang/actions/runs/31297749345)
- **分支**: `amd/mm-gen-nested-b1-harness`
- **总耗时**: 32.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31297749345

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 21.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传工件时未找到失败文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31297749345/job/93205520817) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最终上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本或测试未实际运行，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31297749345/job/93205520817

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31297749345/job/93205520819) |


## [Run #31296907910](https://github.com/sgl-project/sglang/actions/runs/31296907910)
- **分支**: `kp/dcp-from-parallel-state`
- **总耗时**: 58.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31296907910

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 25.2min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428064) |
| base-b-test-16-npu-a3 / run (0) | 40.3min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428100) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 27.4min | 精度回归 | NPU精度测试未达基线，Qwen3.5-9B GSM8K准确率0.8低于基线0.835。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428284) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | 性能测试未达到基线，测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93204953321) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.2min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93206537673) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.2min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93208012406) |

- **multimodal-gen-test-2-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（未找到文件）等常规信息，未出现测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428064

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，3/6测试通过，该用例耗时365秒后失败，属于功能测试未通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428100

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py失败，准确率0.8低于基线0.835，3个测试全部未通过，可能因模型或代码改动导致精度下降。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428284

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms的吞吐量为395.6，低于基线390.5859，导致测试失败，退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93204953321

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现其他三个作业（multimodal-gen-test-2-npu-a3、base-c-test-acc-2-npu-a3、base-c-test-perf-8-npu-a3）失败，触发fast-fail机制，本作业未实际运行即被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93206537673

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业未实际运行，因PR健康检查检测到其他4个根因作业失败（如multimodal-gen-test-2-npu-a3等），触发fast-fail机制，本作业被跳过，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93208012406

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428086) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428101) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428108) |
| base-b-test-4-npu-a3 / run (0) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428111) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428113) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428120) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428138) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428257) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428290) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296907910/job/93203428297) |


## [Run #31296878038](https://github.com/sgl-project/sglang/actions/runs/31296878038)
- **分支**: `cheng/gc-rc-review`
- **总耗时**: 45.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31296878038

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 19.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358135) |
| multimodal-gen-test-1-npu-a3 | 44.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358175) |
| base-b-test-16-npu-a3 / run (0) | 36.5min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358213) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 38.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358333) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93204415698) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.2min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93206011044) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.9min | 环境问题 | 自定义容器执行失败，pip升级过程中被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93207216411) |

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行阶段的输出，只有GitHub Actions的初始化、上传artifact（无文件）和清理步骤。无法判断具体失败原因，可能为日志截断或作业在测试前已异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358135

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures目录（无文件），未包含multimodal-gen测试执行过程或失败断言，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358175

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试返回退出码1，导致作业失败。该测试属于pd_disaggregation功能模块，可能涉及代码逻辑或环境配置问题，需进一步查看具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358213

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现"Executing the custom container implementation failed"错误，表明自托管runner的容器执行环境出现问题，导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358333

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量379.23，低于基线390.5859，性能回归约2.9%，未通过性能阈值检查，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93204415698

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示在安装Rust工具链后，执行自定义容器实现时失败，错误提示需联系自托管runner管理员，属于基础设施环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93206011044

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在安装pip时，卸载旧版本pip后执行失败，报错'Executing the custom container implementation failed'，导致容器无法继续运行，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93207216411

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358183) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358186) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358193) |
| base-b-test-2-npu-a3 / run (0) | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358219) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358220) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358240) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358324) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358326) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31296878038/job/93203358345) |


## [Run #31295947869](https://github.com/sgl-project/sglang/actions/runs/31295947869)
- **分支**: `feat/sol-attn`
- **总耗时**: 30.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31295947869

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 29.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31295947869/job/93200982977) |
| multimodal-gen-test-2-npu-a3 | 9.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31295947869/job/93200982991) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31295947869/job/93200982977

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅显示上传diffusion-failures工件时无文件，可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31295947869/job/93200982991


## [Run #31295308002](https://github.com/sgl-project/sglang/actions/runs/31295308002)
- **分支**: `feat/rollout-weight-sync-own-gpu-payload`
- **总耗时**: 14.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31295308002

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 13.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31295308002/job/93202621578) |

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业整体失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31295308002/job/93202621578

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31295308002/job/93202621841) |


## [Run #31293486179](https://github.com/sgl-project/sglang/actions/runs/31293486179)
- **分支**: `cheng/gc-rc-review`
- **总耗时**: 85.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31293486179

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 34.4min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748368) |
| multimodal-gen-test-2-npu-a3 | 21.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748371) |
| base-b-test-16-npu-a3 / run (0) | 35.2min | 代码错误 | NPU PD分离测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748426) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.8min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，2/3 通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748555) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查中lint检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93195283674) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 3.7min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93196972158) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93198447564) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node.js 20弃用警告和上传artifact时无文件提示，无法判断实际失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748368

- **multimodal-gen-test-2-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传artifact时未找到diffusion-failures目录，说明测试可能未产生失败文件，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748371

- **base-b-test-16-npu-a3 / run (0)**: 测试test_npu_pd_disaggregation.py返回退出码1，6个测试中3个通过3个失败，该用例耗时375秒后失败，属于功能测试未通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748426

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码 1，耗时 1320 秒，未达预期精度标准，其余两个测试通过，判定为精度回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748555

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段执行health-check时，lint检查结论为failure，触发fast-fail机制，作业未进入实际测试即退出，属于前置检查失败，非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93195283674

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93196972158

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行健康检查时，lint 检查结论为 failure，触发了 fast-fail 机制，导致脚本以非零码退出，作业未能进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93198447564

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748399) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748401) |
| base-b-test-4-npu-a3 / run (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748422) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748423) |
| base-b-test-4-npu-a3 / run (1) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748425) |
| base-b-test-2-npu-a3 / run (0) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748428) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748550) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748552) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293486179/job/93194748593) |


## [Run #31293137640](https://github.com/sgl-project/sglang/actions/runs/31293137640)
- **分支**: `dev/dlal/norm-quant-fusion-runtime`
- **总耗时**: 83.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31293137640

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 20.8min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945277) |
| base-b-test-16-npu-a3 / run (0) | 38.9min | 代码错误 | NPU PD分离测试失败，3/6测试通过，1个测试文件返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945364) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 25.1min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946363) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 26.4min | 性能回归 | 性能测试未达基线，吞吐量低于预期导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946399) |

- **multimodal-gen-test-2-npu-a3**: 日志仅显示GitHub Actions初始化、Node版本警告及上传artifact步骤，未包含实际测试命令或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945277

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败，耗时349秒，返回退出码1。其他3个测试均通过，表明是特定测试用例的功能性问题，而非环境或超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945364

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例TestNPUMiniMaxM2_5W8A8_4P_In64k_Out1k_Prefix90_50ms实际吞吐量387.82，低于基线390.5859，未通过性能阈值，测试脚本返回退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946363

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: TestQwen235B 用例实测吞吐量 5941.24，低于基线 6189.0，未通过性能阈值检查，测试脚本返回退出码 1，最终 0/4 用例通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946399

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945601) |
| base-b-test-4-npu-a3 / run (1) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945668) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945684) |
| base-b-test-4-npu-a3 / run (0) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945695) |
| multimodal-gen-test-1-npu-a3 | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945716) |
| base-b-test-1-npu-a3 / run (0) | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945718) |
| base-b-test-2-npu-a3 / run (0) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207945841) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946104) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946141) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946167) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946187) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946323) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 79.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31293137640/job/93207946335) |


---
*Auto-generated by npu_pr_monitor.py*