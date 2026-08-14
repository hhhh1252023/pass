# NPU CI 执行监控
**生成时间**: 2026-08-14 12:07 UTC
**分析 Run 数**: 11

---

## 📊 本次执行总结

- **成功 Job 数**: 73
- **失败 Run 数**: 11
- **成功 Job 平均耗时**: 32.0min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 264.5min | #31765061581 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94702249412) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 143.2min | #31770426395 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082950) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.3min | #31765760015 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273841) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.5min | #31766164503 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478505) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.2min | #31766209603 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724604) |
| base-b-test-16-npu-a3 / run (0) | 63.0min | #31765061581 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659148960) |
| base-b-test-16-npu-a3 / run (0) | 60.6min | #31766164503 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478281) |
| base-b-test-16-npu-a3 / run (0) | 57.4min | #31766209603 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724442) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 56.0min | #31769795562 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229403) |
| base-b-test-16-npu-a3 / run (0) | 51.9min | #31769795562 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229152) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31765061581<br>[#34277 [DSV4] Emit TMA-aligned UE8M0 scales for FP8 einsum](https://github.com/sgl-project/sglang/pull/34277) | `dsv4/pack-tma-for-einsum` | 533.6min | 2 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31765061581) |
| #31766209603<br>[#32500 feat(hicache): support Ascend Mamba states with FIA and async IO](https://github.com/sgl-project/sglang/pull/32500) | `feat/ascend-hicache-mamba-fia-async` | 437.6min | 5 | base-b-test-8-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31766209603) |
| #31770426395<br>[#24911 Profiling Enhancements [2/3]: detailed execution step annotations](https://github.com/sgl-project/sglang/pull/24911) | `feat/roofline_annotations` | 433.6min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31770426395) |
| #31769795562<br>[#34798 [HiCache] Buffer-only mode for HiCache host memory layer](https://github.com/sgl-project/sglang/pull/34798) | `hicache-buffer-only-mode` | 411.3min | 4 | base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31769795562) |
| #31766164503<br>[#34492 XPU: SGLANG_USE_SGL_XPU default to true](https://github.com/sgl-project/sglang/pull/34492) | `SGLANG_USE_SGL_XPU` | 409.5min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31766164503) |
| #31765760015<br>[#30345 [Intel][XPU][LoRA] Enable LoRA on Intel XPU](https://github.com/sgl-project/sglang/pull/30345) | `enable-lora-xpu` | 381.3min | 9 | base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31765760015) |
| #31766142242<br>[#33998 [HiCache] Optimize LogicalHostPool free-list release](https://github.com/sgl-project/sglang/pull/33998) | `hicache-logical-host-pool-wwm` | 374.1min | 7 | base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31766142242) |
| #31768109441<br>[#34406 TP/PP Consensus checker](https://github.com/sgl-project/sglang/pull/34406) | `consensus_checker_0806` | 345.7min | 9 | base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31768109441) |
| #31769555791<br>[#32898 [PD] Fix reasoning token accounting for the handoff token](https://github.com/sgl-project/sglang/pull/32898) | `fix-pd-handoff-reasoning-tokens` | 306.8min | 9 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31769555791) |
| #31770361131<br>[#34801 [PD] Preserve decode KV across retraction in HiCache](https://github.com/sgl-project/sglang/pull/34801) | `shiyang/pd-host-pool-retraction-backup` | 304.2min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31770361131) |
| #31771618565<br>[#33676 [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management](https://github.com/sgl-project/sglang/pull/33676) | `main_8.5` | 274.4min | 11 | base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31771618565) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 11 | 10 | 0 | 1 |
| base-b-test-1-npu-a3 / run (0) | 11 | 6 | 0 | 5 |
| base-b-test-16-npu-a3 / run (0) | 11 | 4 | 0 | 7 |
| base-b-test-2-npu-a3 / run (0) | 11 | 4 | 0 | 7 |
| base-b-test-4-npu-a3 / run (0) | 11 | 6 | 0 | 5 |
| base-b-test-4-npu-a3 / run (1) | 11 | 5 | 0 | 6 |
| base-b-test-8-npu-a3 / run (0) | 11 | 3 | 0 | 8 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11 | 4 | 1 | 6 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11 | 4 | 0 | 7 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11 | 7 | 0 | 4 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11 | 7 | 0 | 4 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 4 | 1 | 0 | 3 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 4 | 0 | 0 | 4 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 7 | 2 | 0 | 5 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 7 | 0 | 0 | 7 |
| multimodal-gen-test-1-npu-a3 | 10 | 10 | 0 | 0 |

---


## [Run #31771618565](https://github.com/sgl-project/sglang/actions/runs/31771618565)
- **分支**: `main_8.5`
- **总耗时**: 274.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31771618565

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 0.9min | 其他 | 健康检查级联失败，根因是其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648530) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648573) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648581) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业根因失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648592) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | 代码错误 | NPU注意力测试test_npu_ascend_dsv4_backend.py失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648601) |
| base-b-test-16-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648603) |
| base-b-test-4-npu-a3 / run (1) | 0.7min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648628) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648662) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648712) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648738) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648764) |

- **base-b-test-4-npu-a3 / run (0)**: 日志显示本作业因健康检查过滤级联失败而被快速失败，根因是base-a-test-1-npu-a2作业失败，本作业并非直接失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648530

- **base-b-test-8-npu-a3 / run (0)**: 该作业在启动前的健康检查中发现根因失败作业base-a-test-1-npu-a2，触发快速失败机制，本作业被跳过未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648573

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业base-a-test-1-npu-a2，本作业因级联失败被过滤并快速失败，非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648581

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段发现根因失败作业base-a-test-1-npu-a2，触发fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648592

- **base-a-test-1-npu-a2 / run (0)**: 测试test_npu_ascend_dsv4_backend.py执行失败（退出码1），而test_npu_ascend_backend.py通过。可能是DSV4后端相关代码存在bug或兼容性问题，需检查该测试文件及对应实现。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648601

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是base-a-test-1-npu-a2作业失败，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648603

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是base-a-test-1-npu-a2 / run (0)失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648628

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因作业base-a-test-1-npu-a2失败，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648662

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业base-a-test-1-npu-a2，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648712

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动前健康检查发现根因作业base-a-test-1-npu-a2失败，触发fast-fail机制，本作业未实际运行即被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648738

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业在启动前被健康检查脚本判定为级联失败，根因是base-a-test-1-npu-a2作业失败，本作业被快速失败机制跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648764

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31771618565/job/94685648451) |


## [Run #31770426395](https://github.com/sgl-project/sglang/actions/runs/31770426395)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 433.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31770426395

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 4.0min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082748) |
| base-b-test-4-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082758) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082771) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082818) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082880) |
| base-b-test-2-npu-a3 / run (0) | 2.0min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082894) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675083022) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675083103) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94732935525) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94757525593) |

- **base-b-test-1-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082748

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到base-b-test-1和base-b-test-2作业为根因失败，本作业被判定为级联失败并快速跳过，并非自身测试失败，属于环境或上游问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082758

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到base-b-test-1和base-b-test-2两个根因作业失败，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082771

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现base-b-test-1和base-b-test-2作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082818

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是base-b-test-1-npu-a3和base-b-test-2-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082880

- **base-b-test-2-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业失败。属于临时网络或缓存服务问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082894

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动前的PR健康检查中，检测到base-b-test-1-npu-a3和base-b-test-2-npu-a3两个根因作业已失败，因此本作业被快速失败机制跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675083022

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查发现base-b-test-1和base-b-test-2两个根因作业失败，触发了fast-fail机制，本作业未实际运行即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675083103

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到多个根因失败作业（base-b-test-1/2-npu-a3），本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94732935525

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是base-b-test-1-npu-a3和base-b-test-2-npu-a3的run作业失败，导致本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94757525593

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082667) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082716) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 143.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082950) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770426395/job/94675082968) |


## [Run #31770361131](https://github.com/sgl-project/sglang/actions/runs/31770361131)
- **分支**: `shiyang/pd-host-pool-retraction-backup`
- **总耗时**: 304.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31770361131

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他作业根因失败，导致本作业被快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888145) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888172) |
| base-b-test-1-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888193) |
| base-b-test-16-npu-a3 / run (0) | 1.4min | 其他 | 级联失败，根因在其他作业 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888214) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888222) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888322) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.6min | 其他 | 健康检查快速失败，因同批次其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888344) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888383) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 环境问题 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888387) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.8min | 精度回归 | NPU精度测试用例qwen3_5_9b_bf16_1p_gsm8k执行失败，0/3测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888608) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业base-c-test-acc-2-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身测试失败，属于环境或上游作业问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888145

- **base-b-test-4-npu-a3 / run (0)**: 日志显示本作业在健康检查阶段因检测到根因作业base-c-test-acc-2-npu-a3失败而触发fast-fail机制，属于级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888172

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是base-c-test-acc-2-npu-a3作业失败，本作业因快速失败机制被跳过，并非自身代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888193

- **base-b-test-16-npu-a3 / run (0)**: 本作业因健康检查检测到根因作业base-c-test-acc-2-npu-a3失败而被快速跳过，属于级联失败，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888214

- **base-b-test-4-npu-a3 / run (1)**: 健康检查显示根因作业base-c-test-acc-2-npu-a3失败，本作业作为级联失败被过滤，最终因快速失败策略终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888222

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业为base-c-test-acc-2-npu-a3，本作业因快速失败（Fast-fail）被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888322

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-acc-2-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于CI流程的级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888344

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业base-c-test-acc-2-npu-a3，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888383

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是base-c-test-acc-2-npu-a3作业失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888387

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，运行83秒后失败，所有3个测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888608

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888072) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770361131/job/94674888101) |


## [Run #31769795562](https://github.com/sgl-project/sglang/actions/runs/31769795562)
- **分支**: `hicache-buffer-only-mode`
- **总耗时**: 411.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31769795562

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 1.9min | 环境问题 | Rust工具链下载超时导致CI失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229151) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 133.8min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229383) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94728433923) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 35.3min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，未达到性能目标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94738862067) |

- **base-b-test-8-npu-a3 / run (0)**: 在安装Rust 1.92时，从内部缓存服务下载channel-rust-1.92.toml.sha256文件超时，导致rustup初始化失败，进而使整个作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229151

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 退出码为1，而其他两个测试通过，表明该模型精度未达标，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229383

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1141秒后失败，该测试为性能测试，要求50ms延迟，可能因性能未达标或环境问题导致退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94728433923

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件4个用例中1个失败，qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例退出码1，耗时1307秒，未通过性能基准，其余3个用例通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94738862067

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229013) |
| base-b-test-4-npu-a3 / run (0) | 31.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229076) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229083) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229085) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229142) |
| base-b-test-16-npu-a3 / run (0) | 51.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229152) |
| base-b-test-4-npu-a3 / run (1) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229177) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229293) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 56.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229403) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94673229461) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769795562/job/94731665078) |


## [Run #31769555791](https://github.com/sgl-project/sglang/actions/runs/31769555791)
- **分支**: `fix-pd-handoff-reasoning-tokens`
- **总耗时**: 306.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31769555791

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 2.4min | 环境问题 | Rust工具链下载超时导致CI失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551367) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551397) |
| base-b-test-16-npu-a3 / run (0) | 2.4min | 环境问题 | 健康检查发现多个NPU测试作业级联失败，根因作业为base-b-test-2和base-c-test-acc-2，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551417) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551434) |
| base-b-test-4-npu-a3 / run (1) | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551505) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 作业因其他根因作业失败被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551663) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 2.1min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551717) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551732) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.9min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94721368759) |

- **base-b-test-2-npu-a3 / run (0)**: 在安装Rust 1.92时，从内部缓存服务下载channel-rust-1.92.toml.sha256文件超时，导致rustup初始化失败，属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551367

- **base-b-test-1-npu-a3 / run (0)**: 日志显示health-check检测到base-b-test-2-npu-a3和base-c-test-acc-2-npu-a3两个根因失败作业，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551397

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是base-b-test-2-npu-a3和base-c-test-acc-2-npu-a3两个作业失败，导致当前作业被快速失败跳过，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551417

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到base-b-test-2-npu-a3和base-c-test-acc-2-npu-a3为根因失败，本作业因级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551434

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业（base-b-test-2-npu-a3和base-c-test-acc-2-npu-a3），触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551505

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查发现同PR中base-b-test-2-npu-a3和base-c-test-acc-2-npu-a3两个根因作业失败，触发fast-fail机制，本作业被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551663

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在安装 Rust 工具链时，从内部缓存服务下载 rust-1.92 元数据文件超时，导致脚本退出，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551717

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到base-b-test-2-npu-a3和base-c-test-acc-2-npu-a3两个根因作业失败，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551732

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查过滤级联失败后，根因作业为base-b-test-2-npu-a3和base-c-test-acc-2-npu-a3，本作业因这些根因失败被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94721368759

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551288) |
| base-a-test-1-npu-a2 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551409) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551576) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769555791/job/94672551730) |


## [Run #31768109441](https://github.com/sgl-project/sglang/actions/runs/31768109441)
- **分支**: `consensus_checker_0806`
- **总耗时**: 345.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31768109441

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150311) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150349) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150357) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150368) |
| base-b-test-4-npu-a3 / run (1) | 1.8min | 环境问题 | Rust工具链下载超时导致CI失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150408) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150501) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.3min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150582) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150595) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94727881335) |

- **base-b-test-1-npu-a3 / run (0)**: 作业启动后健康检查发现base-b-test-4-npu-a3作业失败，被判定为根因失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150311

- **base-b-test-16-npu-a3 / run (0)**: 作业在健康检查阶段检测到base-b-test-4-npu-a3 / run (1)为根因失败作业，根据快速失败策略跳过本作业，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150349

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业base-b-test-4-npu-a3/run(1)，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150357

- **base-b-test-2-npu-a3 / run (0)**: 作业启动后，健康检查检测到同一PR中的base-b-test-4-npu-a3作业失败，判定为根因失败，因此本作业被快速失败机制跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150368

- **base-b-test-4-npu-a3 / run (1)**: 在安装Rust 1.92时，从内部缓存服务下载channel-rust-1.92.toml.sha256超时，导致rustup初始化失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150408

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-4-npu-a3/run (1)，触发fast-fail机制，本作业未实际运行即被取消，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150501

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示根因失败作业为base-b-test-4-npu-a3/run(1)，本作业因健康检查过滤级联失败而被快速失败，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150582

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业base-b-test-4-npu-a3失败，本作业因级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150595

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是base-b-test-4-npu-a3/run失败，本作业因fast-fail被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94727881335

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150247) |
| base-a-test-1-npu-a2 / run (0) | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150293) |
| base-b-test-4-npu-a3 / run (0) | 35.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150414) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31768109441/job/94668150563) |


## [Run #31766209603](https://github.com/sgl-project/sglang/actions/runs/31766209603)
- **分支**: `feat/ascend-hicache-mamba-fia-async`
- **总耗时**: 437.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31766209603

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724454) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.9min | 环境问题 | K8s Pod 启动失败，作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724791) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94719721228) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94724199205) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94734376990) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到base-c-test-acc-16-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-8-npu-a3）被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724454

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 Pod linux-aarch64-a3-16-cn12-001-772vk-runner-57gsq-workflow 状态为 Failed，导致作业在初始化阶段即失败，未进入实际测试。属于基础设施环境问题，需联系 runner 管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724791

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业base-c-test-acc-16-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94719721228

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因作业 base-c-test-acc-16-npu-a3 失败，本作业因快速失败（fast-fail）被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94724199205

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因作业 base-c-test-acc-16-npu-a3 失败，本作业作为级联失败被 fast-fail 跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94734376990

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724365) |
| base-b-test-2-npu-a3 / run (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724421) |
| base-b-test-16-npu-a3 / run (0) | 57.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724442) |
| base-b-test-1-npu-a3 / run (0) | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724458) |
| base-a-test-1-npu-a2 / run (0) | 10.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724461) |
| base-b-test-4-npu-a3 / run (0) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724507) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724562) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724604) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724739) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766209603/job/94662724771) |


## [Run #31766164503](https://github.com/sgl-project/sglang/actions/runs/31766164503)
- **分支**: `SGLANG_USE_SGL_XPU`
- **总耗时**: 409.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31766164503

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 超时 | 性能测试用例执行超时或失败，导致作业整体退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94716438234) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94725847153) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94727275050) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94734354437) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1117秒后返回退出码1，未通过，导致作业失败。可能因性能未达标或环境问题，但日志未显示具体错误，需进一步查看测试输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94716438234

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在健康检查阶段因根因作业base-c-test-perf-8-npu-a3失败而触发fast-fail，未实际运行测试，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94725847153

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败机制跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94727275050

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业被级联过滤并快速失败，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94734354437

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478149) |
| base-b-test-4-npu-a3 / run (0) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478224) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478263) |
| base-b-test-16-npu-a3 / run (0) | 60.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478281) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478282) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478295) |
| base-b-test-2-npu-a3 / run (0) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478321) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478505) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 47.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478508) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478529) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478543) |
| base-a-test-1-npu-a2 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766164503/job/94662478547) |


## [Run #31766142242](https://github.com/sgl-project/sglang/actions/runs/31766142242)
- **分支**: `hicache-logical-host-pool-wwm`
- **总耗时**: 374.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31766142242

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 2.8min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424854) |
| base-b-test-8-npu-a3 / run (0) | 0.9min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424876) |
| base-b-test-16-npu-a3 / run (0) | 1.3min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424880) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，由其他根因作业触发级联失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662425115) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662425132) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662425196) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.7min | 环境问题 | 健康检查检测到其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94721863941) |

- **base-b-test-2-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424854

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行PR健康检查时，检测到同一PR中base-b-test-2-npu-a3作业已失败，且被判定为根因失败，因此本作业被快速失败机制跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424876

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为base-b-test-2-npu-a3，本作业（base-b-test-16-npu-a3）被级联跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424880

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业在“Check PR test health”步骤被跳过，因为根因作业base-b-test-2-npu-a3失败，导致本作业被快速失败机制终止，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662425115

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因作业base-b-test-2-npu-a3失败，本作业因快速失败（fast-fail）被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662425132

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-2-npu-a3/run(0)，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662425196

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现根因作业base-b-test-2-npu-a3失败，导致本作业被级联跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94721863941

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424821) |
| base-b-test-4-npu-a3 / run (1) | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424885) |
| base-a-test-1-npu-a2 / run (0) | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424890) |
| base-b-test-4-npu-a3 / run (0) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662424891) |
| base-b-test-1-npu-a3 / run (0) | 27.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662425028) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766142242/job/94662425126) |


## [Run #31765760015](https://github.com/sgl-project/sglang/actions/runs/31765760015)
- **分支**: `enable-lora-xpu`
- **总耗时**: 381.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31765760015

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 2.5min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273313) |
| base-b-test-2-npu-a3 / run (0) | 1.4min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273370) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273371) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273505) |
| base-b-test-8-npu-a3 / run (0) | 3.2min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273813) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | 性能测试未达预期，测试用例执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94705492177) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94709699899) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94714352067) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 环境问题 | 健康检查发现多个根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94723543219) |

- **base-b-test-4-npu-a3 / run (1)**: 在安装 Rust 1.92 时，从内部缓存服务下载 channel-rust-1.92.toml 超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273313

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业（base-b-test-4-npu-a3 / run (1) 和 base-b-test-8-npu-a3 / run (0)），触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273370

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到 base-b-test-4-npu-a3 / run (1) 和 base-b-test-8-npu-a3 / run (0) 两个根因作业失败，触发了 fast-fail 机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273371

- **base-b-test-16-npu-a3 / run (0)**: 作业在健康检查阶段检测到其他3个根因作业失败（base-b-test-4-npu-a3/run(1)、base-b-test-8-npu-a3/run(0)、base-c-test-perf-8-npu-a3），触发fast-fail机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273505

- **base-b-test-8-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本执行失败，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273813

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例 test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py 执行失败，退出码1，耗时1101秒，未通过性能测试要求，可能因性能未达标或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94705492177

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到多个根因失败作业（如base-b-test-4-npu-a3等），本作业因快速失败策略被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94709699899

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业（base-b-test-4-npu-a3等），触发fast-fail机制，本作业未实际运行即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94714352067

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查过滤了级联失败后，根因作业为base-b-test-4-npu-a3/run(1)、base-b-test-8-npu-a3/run(0)和base-c-test-perf-8-npu-a3，本作业因这些根因失败被跳过，非自身代码问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94723543219

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273296) |
| base-a-test-1-npu-a2 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273345) |
| base-b-test-1-npu-a3 / run (0) | 26.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273471) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273715) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 47.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273748) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273783) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765760015/job/94661273841) |


## [Run #31765061581](https://github.com/sgl-project/sglang/actions/runs/31765061581)
- **分支**: `dsv4/pack-tma-for-einsum`
- **总耗时**: 533.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31765061581

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.6min | 精度回归 | NPU精度测试用例qwen3_5_9b_bf16_1p_gsm8k执行失败，0/3测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659149099) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.2min | 性能回归 | NPU性能测试未达预期，测试用例执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94699603533) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，耗时4689秒，超过预估3600秒，所有3个精度测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659149099

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1061秒，未通过性能基准，可能因模型性能下降或环境波动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94699603533

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659148826) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659148872) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659148885) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659148888) |
| base-b-test-4-npu-a3 / run (0) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659148940) |
| base-b-test-16-npu-a3 / run (0) | 63.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659148960) |
| base-b-test-1-npu-a3 / run (0) | 26.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659148982) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659149112) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659149275) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94659149284) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94696318818) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 264.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765061581/job/94702249412) |


---
*Auto-generated by npu_pr_monitor.py*