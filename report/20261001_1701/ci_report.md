# NPU CI 执行监控
**生成时间**: 2026-10-01 09:01 UTC
**分析 Run 数**: 3

---

## 📊 本次执行总结

- **成功 Job 数**: 4
- **失败 Run 数**: 3
- **成功 Job 平均耗时**: 105.0min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| per-commit-16-npu-a3 (0) | 258.9min | #20363118594 | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222553) |
| per-commit-16-npu-a3 (0) | 72.9min | #20366508164 | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831162) |
| per-commit-16-npu-a3 (1) | 47.7min | #20366508164 | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831155) |
| per-commit-16-npu-a3 (1) | 40.5min | #20363118594 | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222565) |

### ❌ 耗时最长的失败 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| per-commit-4-npu-a2 | 47.1min | #20366508164 | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831183) |
| per-commit-4-npu-a2 | 46.0min | #20363118594 | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222549) |
| per-commit-2-npu-a2 (2) | 24.5min | #20363118594 | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222547) |
| per-commit-2-npu-a2 (1) | 24.4min | #20363118594 | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222550) |
| per-commit-2-npu-a2 (0) | 23.6min | #20363118594 | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222560) |
| per-commit-2-npu-a2 (2) | 23.2min | #20366508164 | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831171) |
| per-commit-2-npu-a2 (1) | 23.1min | #20366508164 | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831175) |
| per-commit-2-npu-a2 (0) | 22.3min | #20366508164 | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831228) |
| per-commit-1-npu-a2 | 21.6min | #20363118594 | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222552) |
| per-commit-1-npu-a2 | 21.1min | #20366508164 | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831135) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 | 失败任务链接 |
|----------|----------|------|---------|-------------|------|-------------|
| per-commit-1-npu-a2 | 2 | 0 | 2 | 0 | 0 | [job link 1](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831135)<br>[job link 2](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222552) |
| per-commit-4-npu-a2 | 2 | 0 | 2 | 0 | 0 | [job link 1](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831183)<br>[job link 2](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222549) |
| per-commit-2-npu-a2 (0) | 2 | 0 | 2 | 0 | 0 | [job link 1](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831228)<br>[job link 2](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222560) |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #20363118594<br>[#14164 EP Support for Piecewise Cuda Graph](https://github.com/sgl-project/sglang/pull/14164) | `ep-support` | 274.1min | 5 | per-commit-2-npu-a2 (2), per-commit-4-npu-a2, per-commit-2-npu-a2 (1), per-commit-1-npu-a2, per-commit-2-npu-a2 (0) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/20363118594) |
| #20366508164<br>[#15440 [Quantization][RL] Support Online Blockwise FP8 Quantization](https://github.com/sgl-project/sglang/pull/15440) | `dev/blockwise-fp8-rollout` | 74.4min | 5 | per-commit-1-npu-a2, per-commit-2-npu-a2 (2), per-commit-2-npu-a2 (1), per-commit-4-npu-a2, per-commit-2-npu-a2 (0) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/20366508164) |
| #20369780755<br>[#14091 vlm: Refactor engine vlm params and support precessor output as input](https://github.com/sgl-project/sglang/pull/14091) | `refactor-engine-vlm-params` | 5.2min | 1 | pr-gate / pr-gate | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/20369780755) |

---


## [Run #20369780755](https://github.com/sgl-project/sglang/actions/runs/20369780755)
- **分支**: `refactor-engine-vlm-params`
- **总耗时**: 5.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/20369780755

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| pr-gate / pr-gate | 5.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20369780755/job/58533360414) |

- **pr-gate / pr-gate**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20369780755/job/58533360414


## [Run #20366508164](https://github.com/sgl-project/sglang/actions/runs/20366508164)
- **分支**: `dev/blockwise-fp8-rollout`
- **总耗时**: 74.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/20366508164

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| per-commit-1-npu-a2 | 21.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831135) |
| per-commit-2-npu-a2 (2) | 23.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831171) |
| per-commit-2-npu-a2 (1) | 23.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831175) |
| per-commit-4-npu-a2 | 47.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831183) |
| per-commit-2-npu-a2 (0) | 22.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831228) |

- **per-commit-1-npu-a2**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831135

- **per-commit-2-npu-a2 (2)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831171

- **per-commit-2-npu-a2 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831175

- **per-commit-4-npu-a2**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831183

- **per-commit-2-npu-a2 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831228

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| per-commit-16-npu-a3 (1) | 47.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831155) |
| per-commit-16-npu-a3 (0) | 72.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/20366508164/job/58522831162) |


## [Run #20363118594](https://github.com/sgl-project/sglang/actions/runs/20363118594)
- **分支**: `ep-support`
- **总耗时**: 274.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/20363118594

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| per-commit-2-npu-a2 (2) | 24.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222547) |
| per-commit-4-npu-a2 | 46.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222549) |
| per-commit-2-npu-a2 (1) | 24.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222550) |
| per-commit-1-npu-a2 | 21.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222552) |
| per-commit-2-npu-a2 (0) | 23.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222560) |

- **per-commit-2-npu-a2 (2)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222547

- **per-commit-4-npu-a2**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222549

- **per-commit-2-npu-a2 (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222550

- **per-commit-1-npu-a2**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222552

- **per-commit-2-npu-a2 (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222560

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| per-commit-16-npu-a3 (0) | 258.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222553) |
| per-commit-16-npu-a3 (1) | 40.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/20363118594/job/58512222565) |


---
*Auto-generated by npu_pr_monitor.py*