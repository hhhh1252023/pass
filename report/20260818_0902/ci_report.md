# NPU CI 执行监控
**生成时间**: 2026-08-18 01:02 UTC
**分析 Run 数**: 27

---

## 📊 本次执行总结

- **成功 Job 数**: 151
- **失败 Run 数**: 27
- **成功 Job 平均耗时**: 23.8min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 249.1min | #32053738328 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484080573) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 133.4min | #32057046311 | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640867) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 112.7min | #32045617198 | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432823694) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 94.0min | #32054219060 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732841) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.5min | #32058282822 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473619481) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.5min | #32047166529 | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350870) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.7min | #32053738328 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484078698) |
| base-b-test-16-npu-a3 / run (0) | 55.0min | #32046529863 | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696015) |
| base-b-test-16-npu-a3 / run (0) | 54.6min | #32047166529 | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350511) |
| base-b-test-16-npu-a3 / run (0) | 54.5min | #32053738328 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484077677) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 25 | 0 | 22 | 0 | 3 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25 | 3 | 0 | 21 | 1 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 25 | 6 | 0 | 17 | 2 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21 | 4 | 0 | 16 | 1 |
| base-b-test-16-npu-a3 / run (0) | 25 | 11 | 0 | 13 | 1 |
| base-b-test-2-npu-a3 / run (0) | 25 | 13 | 0 | 12 | 0 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 25 | 11 | 0 | 12 | 2 |
| base-b-test-1-npu-a3 / run (0) | 25 | 12 | 0 | 11 | 2 |
| base-b-test-4-npu-a3 / run (0) | 25 | 12 | 0 | 11 | 2 |
| base-b-test-4-npu-a3 / run (1) | 25 | 13 | 0 | 10 | 2 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 11 | 3 | 0 | 8 | 0 |
| base-b-test-8-npu-a3 / run (0) | 25 | 18 | 0 | 7 | 0 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 6 | 0 | 0 | 6 | 0 |
| base-a-test-1-npu-a2 / run (0) | 25 | 21 | 0 | 4 | 0 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 25 | 21 | 0 | 2 | 2 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 3 | 1 | 0 | 2 | 0 |
| multimodal-gen-test-2-npu-a3 | 1 | 0 | 1 | 0 | 0 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32053738328<br>[#34330 [AMD] Fix weight checking for AITER-shuffled block FP8 weights](https://github.com/sgl-project/sglang/pull/34330) | `pr/aiter-fp8-weight-checker` | 249.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32053738328) |
| #32057046311<br>[#33765 [observability] Attribute DeepGEMM JIT and FlashInfer autotune in the startup breakdown](https://github.com/sgl-project/sglang/pull/33765) | `jason/trace-startup` | 136.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32057046311) |
| #32058922963 | `cheng/gc-s12-carrier` | 128.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32058922963) |
| #32045617198<br>[#34565 [Unified Tree] Support Branching-Point Caching for the SWA Component](https://github.com/sgl-project/sglang/pull/34565) | `cjc/unified-swa-branching` | 116.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32045617198) |
| #32054219060<br>[#35103 [SM12x] Block the DSv4 KV page-mark kernel launch geometry](https://github.com/sgl-project/sglang/pull/35103) | `fix/sm120-page-mark-launch-geometry` | 95.8min | 0 |  | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32054219060) |
| #32058282822<br>[#30575 [AMD] Enable Fast Triton Sparse MLA backend](https://github.com/sgl-project/sglang/pull/30575) | `feat/triton-sparse-mla` | 93.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32058282822) |
| #32047166529<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 87.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32047166529) |
| #32065880698<br>[#35197 fix(kernel) Fix Helion small-token prefill bug](https://github.com/sgl-project/sglang/pull/35197) | `main` | 86.1min | 0 |  | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32065880698) |
| #32077248440<br>[#34299 [KDA] Add zero-copy native prefill checkpoints and packed decode](https://github.com/sgl-project/sglang/pull/34299) | `codex/sglang-phase-a-admission-rebased-20260810` | 74.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32077248440) |
| #32046529863<br>[#35164 Refactor kv cache event mixin into a recorder](https://github.com/sgl-project/sglang/pull/35164) | `kv-events-composition` | 63.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32046529863) |
| #32054390629<br>[#34855 [NPU] [Diffusion] Fix NPU Ring Attention varlen dispatch & restore 2-NPU CI testcase](https://github.com/sgl-project/sglang/pull/34855) | `fix_ring_attention_npu` | 53.7min | 2 | multimodal-gen-test-2-npu-a3, multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32054390629) |
| #32075444744<br>[#34316 [metrics] Fix prefill FLOPs estimate to count prefix and per-request causal pairs](https://github.com/sgl-project/sglang/pull/34316) | `main` | 48.5min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32075444744) |
| #32053258998<br>[#34923 Apply latest DeepEP branch](https://github.com/sgl-project/sglang/pull/34923) | `codex/deepep-nvshmem-qp-depth` | 31.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32053258998) |
| #32067096564<br>[#35126 [Spec] Stage EAGLE draft-extend graph inputs before the verify launch](https://github.com/sgl-project/sglang/pull/35126) | `lsyin/draft-extend-input-staging` | 29.5min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32067096564) |
| #32070946718<br>[#35062 [Misc] Clean up python/sglang package structure](https://github.com/sgl-project/sglang/pull/35062) | `main` | 24.3min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32070946718) |
| #32079908800<br>[#35028 config: one control-plane log for the process](https://github.com/sgl-project/sglang/pull/35028) | `main` | 21.5min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32079908800) |
| #32045916522<br>[#34612 [Diffusion]  Use current_platform instead of hardcoded "cuda" in cosmos3 guardrails ](https://github.com/sgl-project/sglang/pull/34612) | `cosmos3_guardrails_npu` | 18.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32045916522) |
| #32069679049<br>[#35126 [Spec] Stage EAGLE draft-extend graph inputs before the verify launch](https://github.com/sgl-project/sglang/pull/35126) | `lsyin/draft-extend-input-staging` | 16.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32069679049) |
| #32073456535<br>[#35126 [Spec] Stage EAGLE draft-extend graph inputs before the verify launch](https://github.com/sgl-project/sglang/pull/35126) | `lsyin/draft-extend-input-staging` | 16.5min | 3 | multimodal-gen-test-1-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32073456535) |
| #32072021253<br>[#35126 [Spec] Stage EAGLE draft-extend graph inputs before the verify launch](https://github.com/sgl-project/sglang/pull/35126) | `lsyin/draft-extend-input-staging` | 13.5min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32072021253) |
| #32071077797<br>[#35126 [Spec] Stage EAGLE draft-extend graph inputs before the verify launch](https://github.com/sgl-project/sglang/pull/35126) | `lsyin/draft-extend-input-staging` | 11.2min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32071077797) |
| #32069637609<br>[#35198 [Spec] Relay ngram accept tokens through the FutureMap](https://github.com/sgl-project/sglang/pull/35198) | `lsyin/ngram-accept-relay` | 9.3min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32069637609) |
| #32079062719<br>[#34926 Clean deprecated DeepSeek V4 Environs](https://github.com/sgl-project/sglang/pull/34926) | `main` | 9.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32079062719) |
| #32070647400<br>[#35198 [Spec] Relay ngram accept tokens through the FutureMap](https://github.com/sgl-project/sglang/pull/35198) | `main` | 7.5min | 7 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32070647400) |
| #32074765562<br>[#35126 [Spec] Stage EAGLE draft-extend graph inputs before the verify launch](https://github.com/sgl-project/sglang/pull/35126) | `lsyin/draft-extend-input-staging` | 6.5min | 8 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32074765562) |
| #32071418034 | `model-serve_pr/Mamba_2_and_1` | 5.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32071418034) |
| #32064673365<br>[#31453 [Diffusion][Refactor] Refactor and extract complex RoPE implementation to layers/rotary_embedding for MOVA DiT](https://github.com/sgl-project/sglang/pull/31453) | `rope_mova_unification` | 5.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32064673365) |

---


## [Run #32079908800](https://github.com/sgl-project/sglang/actions/runs/32079908800)
- **分支**: `main`
- **总耗时**: 21.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32079908800

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 20.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832368) |
| base-b-test-16-npu-a3 / run (0) | 20.7min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832372) |
| base-b-test-4-npu-a3 / run (1) | 20.7min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832430) |
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业未显示实际测试失败，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832492) |
| base-b-test-1-npu-a3 / run (0) | 20.4min | 环境问题 | 自定义容器执行失败，NPU图捕获过程中容器崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832530) |
| base-b-test-4-npu-a3 / run (0) | 10.6min | 代码错误 | HiCache MLA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832575) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832762) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832766) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.5min | 精度回归 | NPU精度测试用例失败，GLM5模型GSM8K测试未通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832810) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95542130849) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试运行到18%时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是NPU环境或容器配置问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832368

- **base-b-test-16-npu-a3 / run (0)**: 日志显示作业在加载模型分片（约70%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832372

- **base-b-test-4-npu-a3 / run (1)**: 作业在加载模型分片（31%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832430

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures目录时无文件，可能测试未运行或全部通过，但作业被标记失败，需查看更早日志确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832492

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在NPU decode图捕获阶段（bs=40时）容器实现执行失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832530

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试文件在NPU上运行失败，耗时281秒，测试结果为0/5通过。具体失败原因需查看该测试文件的详细输出，可能是功能实现或测试断言问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832575

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于NPU自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832762

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3两个根因失败作业，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832766

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，属于精度回归问题，可能由模型权重或推理逻辑变更导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832810

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3两个根因作业失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95542130849

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832367) |
| base-b-test-8-npu-a3 / run (0) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832463) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32079908800/job/95540832812) |


## [Run #32079062719](https://github.com/sgl-project/sglang/actions/runs/32079062719)
- **分支**: `main`
- **总耗时**: 9.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32079062719

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 7.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392482) |
| base-b-test-16-npu-a3 / run (0) | 5.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392529) |
| base-b-test-4-npu-a3 / run (1) | 7.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392532) |
| base-b-test-2-npu-a3 / run (0) | 7.3min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392541) |
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 作业未显示实际测试失败，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392567) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392586) |
| base-b-test-4-npu-a3 / run (0) | 7.7min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392622) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392801) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.5min | 精度回归 | NPU精度测试用例glm5_top64_pruned_bf16_8p_gsm8k失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392848) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392859) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95539738777) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中（38%进度）时出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392482

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载到31%时，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392529

- **base-b-test-4-npu-a3 / run (1)**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392532

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重（56%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392541

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤因diffusion-failures目录无文件而跳过，未出现测试失败或错误信息，可能为作业提前结束或测试未执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392567

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试运行约7分钟后，在Expert Balancedness日志输出期间，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392586

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载模型分片后，TP各进程获取ASCEND_OPP_PATH时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392622

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392801

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py在44.92秒内失败，0/1通过，属于精度回归问题，可能由模型权重或代码变更导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392848

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在加载模型分片时，GitHub Actions 报错“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于容器环境异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392859

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业（multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3），触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95539738777

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392685) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32079062719/job/95538392868) |


## [Run #32077248440](https://github.com/sgl-project/sglang/actions/runs/32077248440)
- **分支**: `codex/sglang-phase-a-admission-rebased-20260810`
- **总耗时**: 74.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32077248440

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034187) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 精度回归 | NPU精度测试用例失败，GLM5模型GSM8K测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034616) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 70.9min | 超时 | NPU精度测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034632) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95534660323) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | PR健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95542188325) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node.js弃用警告及上传artifact步骤（未找到文件），未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034187

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，耗时45秒，属于精度回归问题，可能由模型权重或代码改动引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034616

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py运行3999秒超过预估3600秒，被强制终止，0/3测试通过，属于超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034632

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95534660323

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查显示multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3为根因失败，本作业因级联失败被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95542188325

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034171) |
| base-b-test-8-npu-a3 / run (0) | 8.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034208) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034211) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034232) |
| base-b-test-4-npu-a3 / run (0) | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034297) |
| base-b-test-16-npu-a3 / run (0) | 45.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034363) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034379) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034641) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32077248440/job/95533034709) |


## [Run #32075444744](https://github.com/sgl-project/sglang/actions/runs/32075444744)
- **分支**: `main`
- **总耗时**: 48.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32075444744

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.6min | 其他 | 作业未显示实际测试失败原因，仅上传失败产物时提示无文件，可能测试未运行或提前退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591075) |
| base-b-test-16-npu-a3 / run (0) | 44.3min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591229) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591273) |
| base-b-test-1-npu-a3 / run (0) | 46.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591389) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 精度回归 | NPU精度测试用例失败，GLM5模型GSM8K测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591743) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591771) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 42.9min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591805) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 环境问题 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95529829678) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行细节，仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本或作业在测试前已中断，需查看完整日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591075

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型权重分片（约9%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591229

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591273

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在运行第9个测试时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是容器环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591389

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，耗时45.51秒，属于精度回归问题，可能由模型权重或代码改动引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591743

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动前的健康检查中检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591771

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在23:07:05时出现"Executing the custom container implementation failed"错误，导致作业中断。这是自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591805

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3，本作业被级联跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95529829678

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591220) |
| base-b-test-4-npu-a3 / run (1) | 28.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591276) |
| base-b-test-8-npu-a3 / run (0) | 13.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591322) |
| base-b-test-2-npu-a3 / run (0) | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591323) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32075444744/job/95527591673) |


## [Run #32074765562](https://github.com/sgl-project/sglang/actions/runs/32074765562)
- **分支**: `lsyin/draft-extend-input-staging`
- **总耗时**: 6.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32074765562

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 2.1min | 其他 | 作业提前结束，未执行实际测试，无失败日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637155) |
| base-a-test-1-npu-a2 / run (0) | 1.8min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637247) |
| base-b-test-4-npu-a3 / run (0) | 4.8min | 环境问题 | Git fetch 过程中 shallow 文件变化导致首次拉取失败，重试后成功。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637294) |
| base-b-test-4-npu-a3 / run (1) | 4.8min | 环境问题 | Git fetch 因 shallow file 变化失败，重试后成功 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637307) |
| base-b-test-1-npu-a3 / run (0) | 4.8min | 环境问题 | Git fetch 因 shallow file 变化失败后重试成功，但首次失败导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637317) |
| base-b-test-8-npu-a3 / run (0) | 1.5min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637322) |
| base-b-test-16-npu-a3 / run (0) | 4.1min | 环境问题 | GitHub Actions 作业在准备阶段因 Git 仓库初始化异常而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637329) |
| base-b-test-2-npu-a3 / run (0) | 1.3min | 环境问题 | 自定义容器执行失败，环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637338) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.2min | 环境问题 | Git 仓库锁文件冲突导致 checkout 失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637472) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.7min | 环境问题 | 自定义容器执行失败，导致作业在安装Rust工具链时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637495) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.5min | 环境问题 | 作业在准备阶段因GitHub Actions runner环境问题失败，未能进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637574) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.6min | 环境问题 | Git 仓库锁文件冲突导致 checkout 失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637609) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入清理流程，未运行多模态生成测试，仅上传了不存在的diffusion-failures目录，可能因前置条件未满足或作业被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637155

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在安装Rust工具链后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637247

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 git fetch 时出现 'fatal: shallow file has changed since we read it' 错误，这是并发或缓存导致的临时性 git 仓库状态问题，重试后成功，属于环境不稳定。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637294

- **base-b-test-4-npu-a3 / run (1)**: 首次 git fetch 报错 'fatal: shallow file has changed since we read it'，退出码128，等待19秒后重试成功，属于临时性环境/网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637307

- **base-b-test-1-npu-a3 / run (0)**: 日志显示首次 git fetch 报错 'fatal: shallow file has changed since we read it'，重试后成功。这属于 Git 仓库状态不一致的临时环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637317

- **base-b-test-8-npu-a3 / run (0)**: 在安装Rust工具链过程中，rustup-init安装组件时自定义容器实现报错，提示联系self-hosted runner管理员，作业因此失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637322

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 Git 初始化时提示默认分支名 'master' 已弃用，且仓库克隆耗时较长，最终作业在运行自定义脚本前终止，疑似环境配置或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637329

- **base-b-test-2-npu-a3 / run (0)**: 作业在安装依赖后执行自定义容器时失败，报错“Executing the custom container implementation failed”，可能是容器镜像或运行环境配置问题，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637338

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 在 fetch 阶段出现 `.git/shallow.lock` 文件已存在的错误，可能是之前 git 进程异常退出残留锁文件，重试后成功，但首次失败已影响作业稳定性。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637472

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装Rust 1.92工具链过程中，执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637495

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示作业在checkout后运行自定义k8s脚本时中断，仅有Node.js 20弃用警告，无测试执行或错误信息，疑似runner或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637574

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在 git fetch 时出现 `.git/shallow.lock` 文件已存在的错误，提示有另一个 git 进程正在运行或之前崩溃，导致无法完成仓库拉取，最终作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32074765562/job/95525637609


## [Run #32073456535](https://github.com/sgl-project/sglang/actions/runs/32073456535)
- **分支**: `lsyin/draft-extend-input-staging`
- **总耗时**: 16.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32073456535

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740601) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740726) |
| base-b-test-4-npu-a3 / run (0) | 14.6min | 环境问题 | 自托管runner容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740813) |
| base-b-test-1-npu-a3 / run (0) | 14.3min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740815) |
| base-b-test-16-npu-a3 / run (0) | 14.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740902) |
| base-b-test-2-npu-a3 / run (0) | 14.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741035) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741160) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.2min | 环境问题 | 自定义容器启动失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741221) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741389) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95523272953) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740601

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型分片加载到82%时，GitHub Actions报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740726

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试用例TestDPAttentionDP2TP2VLM.test_vlm_generate已成功执行并返回200，但随后runner报错"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740813

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在加载模型权重分片时（Multi-thread loading shards 0%），自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740815

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740902

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741035

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在22:10:23出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741160

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动自定义容器时失败，错误提示'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741221

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试请求均返回200 OK，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741389

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95523272953

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740707) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521740846) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32073456535/job/95521741162) |


## [Run #32072021253](https://github.com/sgl-project/sglang/actions/runs/32072021253)
- **分支**: `lsyin/draft-extend-input-staging`
- **总耗时**: 13.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32072021253

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300117) |
| base-b-test-2-npu-a3 / run (0) | 12.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300320) |
| base-b-test-1-npu-a3 / run (0) | 11.9min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300369) |
| base-b-test-16-npu-a3 / run (0) | 11.7min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300497) |
| base-b-test-4-npu-a3 / run (0) | 11.7min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300603) |
| base-b-test-4-npu-a3 / run (1) | 11.7min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300654) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300719) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.1min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300847) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.8min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300883) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因同PR中其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95518887579) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300117

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试逻辑已正常完成（生成请求返回200），但随后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300320

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动NPU容器后，TokenizerManager初始化过程中自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300369

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试用例test_server_info已通过，但随后出现'Executing the custom container implementation failed'错误，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300497

- **base-b-test-4-npu-a3 / run (0)**: 日志显示容器初始化后，在启动sglang服务时出现“Executing the custom container implementation failed”错误，可能是容器环境配置或资源问题，而非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300603

- **base-b-test-4-npu-a3 / run (1)**: 日志显示执行自定义容器实现时失败，错误信息为“Executing the custom container implementation failed”，这通常是自托管runner环境或容器配置问题，而非代码或测试本身的问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300654

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但21:50:09出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300719

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装evalscope等依赖后，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于容器环境配置或运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300847

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管 runner 管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300883

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95518887579

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 8.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300342) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300354) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32072021253/job/95517300754) |


## [Run #32071418034](https://github.com/sgl-project/sglang/actions/runs/32071418034)
- **分支**: `model-serve_pr/Mamba_2_and_1`
- **总耗时**: 5.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32071418034

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403240) |
| base-b-test-2-npu-a3 / run (0) | 4.6min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403243) |
| base-b-test-1-npu-a3 / run (0) | 4.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403265) |
| base-b-test-4-npu-a3 / run (0) | 4.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403280) |
| base-a-test-1-npu-a2 / run (0) | 4.7min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403281) |
| base-b-test-4-npu-a3 / run (1) | 4.8min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403296) |
| base-b-test-16-npu-a3 / run (0) | 4.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403357) |
| base-b-test-8-npu-a3 / run (0) | 4.7min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403371) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403716) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403734) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403784) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.3min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403916) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本警告及上传artifact时未找到diffusion-failures目录，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403240

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在构建sgl-eval依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403243

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载到50%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境或runner配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403265

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403280

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在安装依赖后执行自定义容器时出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于NPU CI环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403281

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403296

- **base-b-test-16-npu-a3 / run (0)**: 作业在启用6个NPU测试后，执行自定义容器时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403357

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动自定义容器时失败，错误信息为"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403371

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403716

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动自定义容器时失败，错误信息为"Executing the custom container implementation failed"，可能是容器镜像或NPU驱动环境配置问题，导致测试无法正常运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403734

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，并非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403784

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在安装依赖和构建sglang后，执行自定义容器时失败，错误为“Executing the custom container implementation failed”，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071418034/job/95515403916


## [Run #32071077797](https://github.com/sgl-project/sglang/actions/runs/32071077797)
- **分支**: `lsyin/draft-extend-input-staging`
- **总耗时**: 11.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32071077797

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.8min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519450) |
| base-b-test-1-npu-a3 / run (0) | 9.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519802) |
| base-b-test-2-npu-a3 / run (0) | 9.3min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519810) |
| base-b-test-8-npu-a3 / run (0) | 9.5min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519857) |
| base-b-test-4-npu-a3 / run (0) | 9.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519934) |
| base-b-test-16-npu-a3 / run (0) | 9.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519964) |
| base-b-test-4-npu-a3 / run (1) | 8.6min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520012) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.9min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520448) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.9min | 精度回归 | NPU精度测试中moonshotai_moonlight_16b_a3b模型GSM8K测试失败，0/3用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520528) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.9min | 精度回归 | NPU精度测试失败，glm5_top64_pruned测试用例未通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520546) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95516302610) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件（未找到文件）等步骤，未展示实际测试执行和失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519450

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载到75%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是容器环境或NPU资源问题导致测试中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519802

- **base-b-test-2-npu-a3 / run (0)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519810

- **base-b-test-8-npu-a3 / run (0)**: 日志显示模型权重加载成功，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于NPU容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519857

- **base-b-test-4-npu-a3 / run (0)**: 日志显示服务已成功启动，但随后出现'Executing the custom container implementation failed'错误，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519934

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型权重时容器执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519964

- **base-b-test-4-npu-a3 / run (1)**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员。日志显示模型加载过程中出现导入错误和内存问题，最终容器崩溃，属于NPU环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520012

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载模型分片后，TP各进程获取ASCEND_OPP_PATH时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520448

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件base-c-test-acc-2-npu-a3中，moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k.py返回退出码1，所有3个测试均未通过，耗时45秒，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520528

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，耗时45秒，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520546

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3、base-c-test-acc-2-npu-a3和base-c-test-acc-16-npu-a3三个根因作业失败，因此本性能测试作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95516302610

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514519877) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32071077797/job/95514520532) |


## [Run #32070946718](https://github.com/sgl-project/sglang/actions/runs/32070946718)
- **分支**: `main`
- **总耗时**: 24.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32070946718

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.8min | 其他 | 作业日志不完整，未显示实际测试执行过程，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990588) |
| base-b-test-1-npu-a3 / run (0) | 19.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990717) |
| base-b-test-16-npu-a3 / run (0) | 18.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990874) |
| base-b-test-4-npu-a3 / run (1) | 19.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990877) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990883) |
| base-b-test-4-npu-a3 / run (0) | 8.5min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990926) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514991261) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514991338) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514991646) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95516328322) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试命令或失败断言，仅有Node版本弃用警告和artifact上传提示（无文件）。可能因日志截断或作业在测试前被取消，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990588

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行第4个测试时，自定义容器实现失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990717

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型分片（约38%）时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990874

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型分片加载完成后，在获取ASCEND_OPP_PATH环境变量时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境配置或容器兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990877

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型权重加载完成后，自定义容器实现执行失败（Executing the custom container implementation failed），提示联系自托管runner管理员，属于NPU测试环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990883

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 执行失败，耗时281秒，0/5测试通过，导致作业整体失败。具体失败原因需查看该测试的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990926

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型权重加载完成后，在获取ASCEND_OPP_PATH环境变量时容器执行失败，错误为'Executing the custom container implementation failed'，属于NPU容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514991261

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514991338

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514991646

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95516328322

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990731) |
| base-b-test-8-npu-a3 / run (0) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514990736) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32070946718/job/95514991427) |


## [Run #32070647400](https://github.com/sgl-project/sglang/actions/runs/32070647400)
- **分支**: `main`
- **总耗时**: 7.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32070647400

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 1.7min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传diffusion-failures产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971377) |
| base-a-test-1-npu-a2 / run (0) | 1.8min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971600) |
| base-b-test-16-npu-a3 / run (0) | 1.6min | 环境问题 | 自定义容器执行失败，rustup安装过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971722) |
| base-b-test-2-npu-a3 / run (0) | 2.1min | 环境问题 | 自定义容器执行失败，下载triton-ascend依赖时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971731) |
| base-b-test-1-npu-a3 / run (0) | 6.1min | 环境问题 | Git 浅克隆时 shallow 文件变化导致首次 fetch 失败，重试后成功。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971783) |
| base-b-test-8-npu-a3 / run (0) | 1.7min | 环境问题 | 自定义容器执行失败，导致作业在安装依赖后中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971830) |
| base-b-test-4-npu-a3 / run (0) | 6.5min | 环境问题 | Git 浅克隆因 shallow 文件变化失败，重试后成功。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971839) |
| base-b-test-4-npu-a3 / run (1) | 5.8min | 环境问题 | Git 浅克隆时因远端仓库强制更新导致 shallow 文件变化，fetch 失败后重试成功。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971890) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.4min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512972377) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 2.6min | 环境问题 | GitHub Actions 工作目录损坏，导致 checkout 失败后自动重建仓库。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512972402) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.9min | 环境问题 | Git 仓库锁文件冲突导致 checkout 失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512972429) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 2.9min | 环境问题 | Git 仓库锁文件冲突导致 checkout 失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512972631) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到具体测试命令和错误输出。仅能确认作业在运行后上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业仍标记为失败，需查看完整日志定位真实原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971377

- **base-a-test-1-npu-a2 / run (0)**: 在安装Rust工具链过程中，rustup-init执行到安装rustc组件时，自定义容器实现报错，提示联系自托管runner管理员，作业因此失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971600

- **base-b-test-16-npu-a3 / run (0)**: 在安装Rust工具链时，下载cargo和clippy组件后，自定义容器实现执行失败，导致作业中止。可能是容器环境或网络问题，与代码无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971722

- **base-b-test-2-npu-a3 / run (0)**: 作业在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中，自定义容器实现执行失败，导致作业终止，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971731

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 git fetch 时出现 'fatal: shallow file has changed since we read it' 错误，这是并发或缓存导致的临时性 git 问题，重试后成功，属于环境不稳定。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971783

- **base-b-test-8-npu-a3 / run (0)**: 日志显示在安装torch-npu和memfabric-zbal后，执行自定义容器时出现错误："Executing the custom container implementation failed"，可能是容器环境或配置问题，需联系runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971830

- **base-b-test-4-npu-a3 / run (0)**: 首次 git fetch 报错 'shallow file has changed since we read it'，可能是并发或缓存问题，重试后成功，属于临时性环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971839

- **base-b-test-4-npu-a3 / run (1)**: checkout 阶段执行 git fetch 时出现 'shallow file has changed since we read it' 错误，首次尝试失败，重试后成功，属于临时性网络或仓库状态问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512971890

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装clang-14时，执行自定义容器实现失败，提示联系self-hosted runner管理员。这属于runner环境或容器配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512972377

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 git 操作报错 'HEAD: unknown revision or path not in the working tree'，工作目录无法清理或重置，最终删除并重新初始化仓库。这属于 runner 环境状态异常，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512972402

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: Git fetch 时 .git/shallow.lock 文件已存在，提示另一个 git 进程正在运行，可能是上次任务残留的锁文件，导致首次 fetch 失败，重试后成功。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512972429

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 在 git fetch 时，.git/shallow.lock 文件已存在，提示另一个 git 进程正在运行，导致首次 fetch 失败。重试后成功，但已造成作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32070647400/job/95512972631


## [Run #32069679049](https://github.com/sgl-project/sglang/actions/runs/32069679049)
- **分支**: `lsyin/draft-extend-input-staging`
- **总耗时**: 16.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32069679049

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026651) |
| base-b-test-2-npu-a3 / run (0) | 15.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026723) |
| base-b-test-16-npu-a3 / run (0) | 14.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026737) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026806) |
| base-b-test-1-npu-a3 / run (0) | 14.2min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026831) |
| base-b-test-4-npu-a3 / run (0) | 14.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026865) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510027222) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.6min | 精度回归 | GLM5 top64 pruned 8P GSM8K 精度测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510027232) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510027239) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95511811735) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026651

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重后，执行自定义容器时失败，报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026723

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在模型加载和初始化完成后，执行自定义容器实现时失败，提示联系自托管runner管理员。可能是NPU设备、容器配置或镜像问题导致环境不可用。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026737

- **base-b-test-4-npu-a3 / run (1)**: 作业在torch_npu初始化阶段失败，日志显示transfer_to_npu警告后容器执行中断，报错'Executing the custom container implementation failed'，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026806

- **base-b-test-1-npu-a3 / run (0)**: 测试在运行第5个用例时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026831

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试服务已成功启动并完成请求，但随后出现'Executing the custom container implementation failed'错误，属于自托管runner容器环境问题，非代码或测试逻辑失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026865

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在执行过程中容器实现失败，提示联系自托管runner管理员，属于runner环境问题而非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510027222

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试 test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py 返回退出码1，0/1测试通过，耗时44.78秒，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510027232

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510027239

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业启动前的健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3两个根因作业失败，因此本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95511811735

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026773) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510026834) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32069679049/job/95510027283) |


## [Run #32069637609](https://github.com/sgl-project/sglang/actions/runs/32069637609)
- **分支**: `lsyin/ngram-accept-relay`
- **总耗时**: 9.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32069637609

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543524) |
| base-b-test-4-npu-a3 / run (1) | 7.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543644) |
| base-b-test-1-npu-a3 / run (0) | 7.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543688) |
| base-b-test-16-npu-a3 / run (0) | 7.2min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543707) |
| base-b-test-2-npu-a3 / run (0) | 7.8min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543753) |
| base-b-test-4-npu-a3 / run (0) | 7.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543816) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543833) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.5min | 精度回归 | NPU精度测试失败，glm5_top64_pruned_bf16_8p_gsm8k测试用例未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510544157) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.3min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510544200) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.8min | 环境问题 | 自定义容器执行失败，模型权重加载过程中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510544416) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95511873073) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543524

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型权重加载完成后，在获取ASCEND_OPP_PATH环境变量时容器执行失败，错误为'Executing the custom container implementation failed'，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543644

- **base-b-test-1-npu-a3 / run (0)**: 日志显示容器在运行torch_npu相关测试时，因自定义容器实现执行失败而终止，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543688

- **base-b-test-16-npu-a3 / run (0)**: 日志显示执行自定义容器实现时出错（Executing the custom container implementation failed），提示联系自托管runner管理员，属于NPU CI环境配置或容器镜像问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543707

- **base-b-test-2-npu-a3 / run (0)**: 日志显示作业在加载模型权重（Multi-thread loading shards）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543753

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在捕获批次（bs=88）时出现"Executing the custom container implementation failed"错误，随后作业清理退出。可能是容器运行环境不稳定或资源限制导致，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543816

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动NPU容器时失败，错误为"Executing the custom container implementation failed"，日志显示服务健康检查返回503，容器未能正常启动，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543833

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，耗时45秒，属于精度回归问题，可能由模型权重或代码变更导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510544157

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在构建Python依赖（如jieba、oss2等）时，runner报错“Executing the custom container implementation failed”，属于自托管runner容器环境异常，非代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510544200

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载Qwen3-VL模型权重（约38%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510544416

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段因健康检查检测到同PR中multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95511873073

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510543746) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32069637609/job/95510544056) |


## [Run #32067096564](https://github.com/sgl-project/sglang/actions/runs/32067096564)
- **分支**: `lsyin/draft-extend-input-staging`
- **总耗时**: 29.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32067096564

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 日志不完整，未显示测试执行过程，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891599) |
| base-b-test-4-npu-a3 / run (0) | 27.9min | 环境问题 | 自定义容器执行失败，NPU后端不支持CUDA相关操作导致运行中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891629) |
| base-b-test-16-npu-a3 / run (0) | 28.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891678) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 27.9min | 环境问题 | 自托管runner执行容器时失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891899) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.2min | 精度回归 | NPU精度测试用例moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k失败，0/3通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891911) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败导致本作业被快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95503540295) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | PR健康检查失败，因其他根因作业失败导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95508124452) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行后上传diffusion-failures目录时提示无文件，未发现明确错误信息，可能测试未执行或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891599

- **base-b-test-4-npu-a3 / run (0)**: 日志显示SymmetricMemory不支持cuda设备类型，且aten::_assert_async算子不支持NPU后端回退到CPU，最终自定义容器实现执行失败，属于NPU环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891629

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型加载到64%时，GitHub Actions报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于NPU容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891678

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但21:10:04出现“Executing the custom container implementation failed”，提示联系runner管理员，属于runner环境或容器执行问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891899

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_moonlight_16b_a3b_bf16_1p_gsm8k.py返回退出码1，耗时34秒，所有3个用例均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891911

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段被健康检查拦截，因同次运行中multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95503540295

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业未实际运行测试，而是在健康检查阶段因检测到其他根因作业（multimodal-gen-test-1-npu-a3和base-c-test-acc-2-npu-a3）失败而被快速失败跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95508124452

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501637359) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891513) |
| base-b-test-4-npu-a3 / run (1) | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891619) |
| base-b-test-2-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891685) |
| base-b-test-8-npu-a3 / run (0) | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891700) |
| base-b-test-1-npu-a3 / run (0) | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891722) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501891872) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32067096564/job/95501892021) |


## [Run #32065880698](https://github.com/sgl-project/sglang/actions/runs/32065880698)
- **分支**: `main`
- **总耗时**: 86.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32065880698

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832718) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 精度回归 | NPU精度测试用例失败，GLM5模型GSM8K测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832764) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95515806827) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832718

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py执行44秒后失败，返回码1，0/1测试通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832764

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-acc-16-npu-a3 作业失败，被判定为根因失败，导致本作业（perf）被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95515806827

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832265) |
| base-b-test-16-npu-a3 / run (0) | 51.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832271) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832286) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832298) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832374) |
| base-b-test-2-npu-a3 / run (0) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832381) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832405) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832705) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95504832808) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32065880698/job/95506006231) |


## [Run #32064673365](https://github.com/sgl-project/sglang/actions/runs/32064673365)
- **分支**: `rope_mova_unification`
- **总耗时**: 5.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32064673365

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32064673365/job/95493932644) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、Node版本警告及上传artifact时未找到diffusion-failures目录。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32064673365/job/95493932644


## [Run #32058922963](https://github.com/sgl-project/sglang/actions/runs/32058922963)
- **分支**: `cheng/gc-s12-carrier`
- **总耗时**: 128.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32058922963

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.6min | 其他 | 作业日志不完整，仅显示上传失败产物和清理过程，未包含实际测试执行结果。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475531847) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 精度回归 | NPU精度测试用例失败，GLM5模型GSM8K测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532586) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 127.5min | 精度回归 | qwen3_5_9b 精度测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532726) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试命令或失败断言，仅显示上传diffusion-failures目录时无文件，可能测试未运行或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475531847

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，属于精度回归问题，可能由模型权重或代码改动引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532586

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k 测试退出码为1，而其他两个测试通过，表明该模型精度未达标，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532726

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532014) |
| base-b-test-16-npu-a3 / run (0) | 53.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532076) |
| base-a-test-1-npu-a2 / run (0) | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532092) |
| base-b-test-2-npu-a3 / run (0) | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532138) |
| base-b-test-8-npu-a3 / run (0) | 9.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532200) |
| base-b-test-4-npu-a3 / run (1) | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532235) |
| base-b-test-4-npu-a3 / run (0) | 31.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532329) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532492) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95475532542) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95477186235) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058922963/job/95487902321) |


## [Run #32058282822](https://github.com/sgl-project/sglang/actions/runs/32058282822)
- **分支**: `feat/triton-sparse-mla`
- **总耗时**: 93.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32058282822

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473618824) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.2min | 精度回归 | NPU精度测试失败，GLM5测试用例未通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473619324) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95475484785) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95485016313) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业 | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95500358097) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含多模态生成测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473618824

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，耗时35秒，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473619324

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段因PR健康检查检测到其他根因作业（multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3）失败而触发fast-fail，本作业未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95475484785

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3两个根因作业失败，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95485016313

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的PR健康检查中检测到其他根因作业（multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3）失败，触发fast-fail机制，本作业被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95500358097

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473387332) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473618858) |
| base-b-test-8-npu-a3 / run (0) | 9.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473618868) |
| base-b-test-1-npu-a3 / run (0) | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473618880) |
| base-b-test-4-npu-a3 / run (0) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473618901) |
| base-b-test-2-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473618935) |
| base-b-test-16-npu-a3 / run (0) | 46.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473618970) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473619046) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473619316) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473619418) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32058282822/job/95473619481) |


## [Run #32057046311](https://github.com/sgl-project/sglang/actions/runs/32057046311)
- **分支**: `jason/trace-startup`
- **总耗时**: 136.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32057046311

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640348) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.5min | 精度回归 | GLM5 top64 pruned 8P GSM8K 精度测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640966) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95471741771) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95481968652) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95508589981) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境准备、Node.js弃用警告及上传diffusion-failures目录时未找到文件，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640348

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试 test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py 返回退出码1，0/1通过，耗时45.54秒，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640966

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3两个根因作业失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95471741771

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业在启动前被PR健康检查拦截，原因是其他根因作业（multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3）失败，触发了fast-fail跳过，本作业未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95481968652

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业（multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3），本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95508589981

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640376) |
| base-b-test-2-npu-a3 / run (0) | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640429) |
| base-b-test-4-npu-a3 / run (0) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640439) |
| base-b-test-1-npu-a3 / run (0) | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640479) |
| base-b-test-16-npu-a3 / run (0) | 53.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640513) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640542) |
| base-b-test-8-npu-a3 / run (0) | 8.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640582) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640863) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 133.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640867) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32057046311/job/95469640872) |


## [Run #32054390629](https://github.com/sgl-project/sglang/actions/runs/32054390629)
- **分支**: `fix_ring_attention_npu`
- **总耗时**: 53.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32054390629

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 4.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462584758) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 51.6min | 精度回归 | NPU精度测试中moonshotai_moonlight_16b_a3b用例失败，导致整体作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585281) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.5min | 精度回归 | NPU精度测试用例glm5_top64_pruned_bf16_8p_gsm8k失败，0/1通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585321) |
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585669) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他相关作业已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95464019758) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.5min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95474243306) |

- **multimodal-gen-test-2-npu-a3**: 日志中只有GitHub Actions的初始化、上传artifact（无文件）和清理步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462584758

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中1/3通过，moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k.py退出码1，运行仅38秒即失败，疑似模型精度或加载问题，需检查该用例日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585281

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，运行45.54秒后失败，属于精度测试未通过，可能为模型精度或数据问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585321

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤。可能因日志截断或作业在测试前已失败，需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585669

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现其他三个作业（multimodal-gen-test-2-npu-a3、base-c-test-acc-16-npu-a3、multimodal-gen-test-1-npu-a3）失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95464019758

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在启动阶段因PR健康检查检测到其他根因作业（multimodal-gen-test-2-npu-a3等）失败而触发fast-fail，本作业本身未执行任何测试，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95474243306

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462584812) |
| base-b-test-16-npu-a3 / run (0) | 51.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462584861) |
| base-b-test-2-npu-a3 / run (0) | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462584873) |
| base-b-test-4-npu-a3 / run (0) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462584911) |
| base-b-test-4-npu-a3 / run (1) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462584941) |
| base-b-test-1-npu-a3 / run (0) | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462584968) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585312) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585339) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054390629/job/95462585407) |


## [Run #32054219060](https://github.com/sgl-project/sglang/actions/runs/32054219060)
- **分支**: `fix/sm120-page-mark-launch-geometry`
- **总耗时**: 95.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32054219060

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95518990249) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.4min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95523627111) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95527645490) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95539661197) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1073秒后失败，该测试为性能测试，预期耗时3600秒，实际提前退出且未通过，表明性能指标未达到要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95518990249

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，根因过滤后仍存在失败作业，因此本作业被快速失败机制跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95523627111

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，将其视为根因，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95527645490

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95539661197

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732124) |
| base-b-test-16-npu-a3 / run (0) | 52.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732206) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732232) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732256) |
| base-b-test-4-npu-a3 / run (0) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732270) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732326) |
| base-b-test-2-npu-a3 / run (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732481) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732719) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732811) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 94.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732841) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32054219060/job/95517732861) |


## [Run #32053738328](https://github.com/sgl-project/sglang/actions/runs/32053738328)
- **分支**: `pr/aiter-fp8-weight-checker`
- **总耗时**: 249.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32053738328

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484076441) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 33.7min | 性能回归 | NPU性能测试中qwen3_6_27b用例失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484080514) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484076441

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 性能测试套件中qwen3_6_27b_w8a8_1p_in64k_out1k_50ms用例执行失败（exit code 1），其余3个用例均通过，表明该模型配置下存在性能不达标或运行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484080514

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 54.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484077677) |
| base-b-test-1-npu-a3 / run (0) | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484077717) |
| base-b-test-4-npu-a3 / run (0) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484077802) |
| base-b-test-4-npu-a3 / run (1) | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484077982) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484078001) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484078052) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484078456) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484078659) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484078698) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484078907) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484079213) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484080566) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 249.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484080573) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053738328/job/95484080660) |


## [Run #32053258998](https://github.com/sgl-project/sglang/actions/runs/32053258998)
- **分支**: `codex/deepep-nvshmem-qp-depth`
- **总耗时**: 31.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32053258998

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566062) |
| base-b-test-8-npu-a3 / run (0) | 3.0min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566230) |
| base-b-test-2-npu-a3 / run (0) | 2.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566242) |
| base-b-test-16-npu-a3 / run (0) | 3.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566260) |
| base-b-test-1-npu-a3 / run (0) | 2.7min | 其他 | 健康检查快速失败，因同一PR中其他作业（base-a-test-1-npu-a2）已失败，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566279) |
| base-a-test-1-npu-a2 / run (0) | 2.0min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566282) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.8min | 精度回归 | GLM5 top64 pruned 8卡 GSM8K 精度测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566641) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566653) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566686) |
| base-b-test-4-npu-a3 / run (1) | 2.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566839) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 2.9min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566993) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566062

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到 base-a-test-1-npu-a2 / run (0) 作业失败，被判定为根因失败，因此本作业（base-b-test-8-npu-a3）被快速失败跳过，并非自身执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566230

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到base-a-test-1-npu-a2作业失败，将其判定为根因，因此本作业（base-b-test-2-npu-a3）被快速失败跳过，并非自身执行出错。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566242

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到另一个作业 base-a-test-1-npu-a2 / run (0) 失败，被判定为根因作业，因此本作业（base-b-test-16-npu-a3）被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566260

- **base-b-test-1-npu-a3 / run (0)**: 日志显示health-check检测到根因失败作业base-a-test-1-npu-a2，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566279

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566282

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试 test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py 返回退出码1，0/1通过，耗时45.74秒，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566641

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业base-a-test-1-npu-a2 / run (0)，触发fast-fail机制，本作业未实际运行即被跳过，属于CI依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566653

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到 `base-a-test-1-npu-a2 / run (0)` 作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566686

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业base-a-test-1-npu-a2 / run (0)，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败的连锁反应。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566839

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动前的健康检查中检测到同一PR的另一个作业base-a-test-1-npu-a2失败，触发了fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566993

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32053258998/job/95457566229) |


## [Run #32047166529](https://github.com/sgl-project/sglang/actions/runs/32047166529)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 87.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32047166529

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350381) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.9min | 精度回归 | NPU精度测试用例失败，GLM5模型GSM8K测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350741) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95521516674) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95531132287) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95539966462) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未展示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350381

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，属于精度回归问题，可能由模型或代码改动引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350741

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3两个根因作业失败，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95521516674

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业因其他根因作业（multimodal-gen-test-1-npu-a3、base-c-test-acc-16-npu-a3）失败而被快速失败跳过，自身未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95531132287

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在健康检查阶段因其他根因作业（multimodal-gen-test-1-npu-a3、base-c-test-acc-16-npu-a3）失败而被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95539966462

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350455) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350473) |
| base-b-test-16-npu-a3 / run (0) | 54.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350511) |
| base-b-test-4-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350527) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350543) |
| base-b-test-2-npu-a3 / run (0) | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350582) |
| base-b-test-4-npu-a3 / run (1) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350626) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 43.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350764) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350800) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32047166529/job/95520350870) |


## [Run #32046529863](https://github.com/sgl-project/sglang/actions/runs/32046529863)
- **分支**: `kv-events-composition`
- **总耗时**: 63.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32046529863

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435695964) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.6min | 精度回归 | NPU精度测试glm5_top64_pruned_bf16_8p_gsm8k失败，0/1通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696325) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 57.8min | 精度回归 | NPU精度测试中moonshotai_moonlight_16b_a3b用例失败，仅1/3通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696330) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435695964

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，耗时45.72秒，未通过精度验证，可能因模型输出与参考不一致导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696325

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件base-c-test-acc-2-npu-a3中，glm4_7_flash用例通过，但moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k.py在35秒内退出码为1，属于精度回归或模型兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696330

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 55.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696015) |
| base-b-test-4-npu-a3 / run (0) | 31.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696067) |
| base-b-test-4-npu-a3 / run (1) | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696068) |
| base-b-test-8-npu-a3 / run (0) | 8.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696090) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696109) |
| base-b-test-1-npu-a3 / run (0) | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696119) |
| base-b-test-2-npu-a3 / run (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696188) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696356) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95435696406) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95437380144) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32046529863/job/95446809563) |


## [Run #32045916522](https://github.com/sgl-project/sglang/actions/runs/32045916522)
- **分支**: `cosmos3_guardrails_npu`
- **总耗时**: 18.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32045916522

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32045916522/job/95433764974) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32045916522/job/95433764974


## [Run #32045617198](https://github.com/sgl-project/sglang/actions/runs/32045617198)
- **分支**: `cjc/unified-swa-branching`
- **总耗时**: 116.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32045617198

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432820859) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.6min | 精度回归 | NPU精度测试glm5_top64_pruned_bf16_8p_gsm8k失败，0/1通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432823719) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 10.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95434448892) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95443206818) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95461601599) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未包含任何测试执行或失败的具体输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432820859

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，耗时44.6秒，未通过精度验证，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432823719

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3两个根因作业失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95434448892

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3为根因失败，本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95443206818

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95461601599

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432822666) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432822673) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432822732) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432822747) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432822809) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432822830) |
| base-b-test-16-npu-a3 / run (0) | 46.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432822879) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 112.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432823694) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432823721) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32045617198/job/95432823725) |


---
*Auto-generated by npu_pr_monitor.py*