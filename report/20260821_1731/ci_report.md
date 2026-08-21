# NPU CI 执行监控
**生成时间**: 2026-08-21 09:31 UTC
**分析 Run 数**: 23

---

## 📊 本次执行总结

- **成功 Job 数**: 102
- **失败 Run 数**: 23
- **成功 Job 平均耗时**: 31.0min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 272.2min | #32117587776 | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95677044654) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.7min | #32138292194 | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717167193) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 103.8min | #32121587014 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195512) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 97.5min | #32131837084 | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697508275) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.0min | #32143584692 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978767) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.3min | #32117587776 | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593879) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.0min | #32144736340 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735772243) |
| base-b-test-16-npu-a3 / run (0) | 75.7min | #32143584692 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978310) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 70.5min | #32142106893 | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727123020) |
| base-b-test-16-npu-a3 / run (0) | 63.7min | #32138292194 | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166768) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 22 | 0 | 22 | 0 | 0 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 22 | 6 | 1 | 14 | 1 |
| base-b-test-4-npu-a3 / run (1) | 22 | 7 | 1 | 13 | 1 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 22 | 7 | 1 | 13 | 1 |
| base-a-test-1-npu-a2 / run (0) | 22 | 9 | 1 | 12 | 0 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22 | 8 | 1 | 12 | 1 |
| base-b-test-4-npu-a3 / run (0) | 22 | 9 | 1 | 11 | 1 |
| base-b-test-16-npu-a3 / run (0) | 22 | 9 | 1 | 11 | 1 |
| base-b-test-8-npu-a3 / run (0) | 22 | 9 | 1 | 11 | 1 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 22 | 9 | 1 | 11 | 1 |
| base-b-test-2-npu-a3 / run (0) | 22 | 10 | 1 | 10 | 1 |
| base-b-test-1-npu-a3 / run (0) | 22 | 10 | 1 | 10 | 1 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 8 | 1 | 1 | 6 | 0 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 6 | 0 | 0 | 6 | 0 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 9 | 4 | 0 | 5 | 0 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 7 | 3 | 0 | 4 | 0 |
| multimodal-gen-test-2-npu-a3 | 1 | 0 | 1 | 0 | 0 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32117587776<br>[#35017 [Scheduler] Add configurable decode interval after prefill](https://github.com/sgl-project/sglang/pull/35017) | `perf/dp-global-prefill-interval` | 386.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32117587776) |
| #32131837084<br>[#33685 [NPU CI] Reorganize test output/log directory structure with workflow context](https://github.com/sgl-project/sglang/pull/33685) | `pllimax/output-log-dir-structure` | 251.9min | 0 |  | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32131837084) |
| #32143584692<br>[#30371 [DSV4] Fix SWA state pool over-allocation by using storage page size instead of model window](https://github.com/sgl-project/sglang/pull/30371) | `dsv4_state_pool_size` | 138.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32143584692) |
| #32138292194<br>[#35238 Exclude multimodal-gen NPU jobs from fast-fail cascade](https://github.com/sgl-project/sglang/pull/35238) | `patch-8` | 135.1min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32138292194) |
| #32143251403<br>[#34881 Stop losing Kimi-K3 tool calls to reasoning, constraint conflicts, and truncation](https://github.com/sgl-project/sglang/pull/34881) | `khoa/fix-required-tool-choice-json-fallback` | 122.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32143251403) |
| #32121587014<br>[#35305 [Kimi-K3] Fix "wrong grids" crash in DP-sharded vision preprocessing](https://github.com/sgl-project/sglang/pull/35305) | `fix/kimi-k3-deferred-grids-global-index` | 115.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32121587014) |
| #32144736340<br>[#35215 [Constrained] Support MistralCommon tokenizers in the XGrammar backend](https://github.com/sgl-project/sglang/pull/35215) | `fix/xgrammar-mistral-common` | 104.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32144736340) |
| #32142106893<br>[#32340 Amd/dsv4 shared experts fusion top6](https://github.com/sgl-project/sglang/pull/32340) | `amd/dsv4-shared-experts-fusion-top6` | 102.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32142106893) |
| #32132517558<br>[#35318 [Perf] PaddleOCR-VL: overlap page preprocessing, pack the ViT, enable prefill CUDA graph](https://github.com/sgl-project/sglang/pull/35318) | `claude/paddleocr-support-optimization-bfaf52` | 92.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32132517558) |
| #32144619391<br>[#35269 [UnifiedTree] feat: support runtime attach/detach for historage](https://github.com/sgl-project/sglang/pull/35269) | `feature/unified-runtime-attach` | 62.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32144619391) |
| #32146312799<br>[#34855 [NPU] [Diffusion] Fix npu diffusion regressions & restore 2-NPU CI testcase](https://github.com/sgl-project/sglang/pull/34855) | `fix_ring_attention_npu` | 57.6min | 2 | multimodal-gen-test-2-npu-a3, multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32146312799) |
| #32146951488<br>[#35318 [Perf] PaddleOCR-VL: overlap page preprocessing, pack the ViT, enable prefill CUDA graph](https://github.com/sgl-project/sglang/pull/35318) | `claude/paddleocr-support-optimization-bfaf52` | 57.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32146951488) |
| #32147191579<br>[#33554 Add new spec-dec support and quant recipe for Nano v3](https://github.com/sgl-project/sglang/pull/33554) | `nemotron-3.5-spec-comparison` | 53.0min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32147191579) |
| #32147234710<br>[#35336 VLM: feed the packed qkv projection output to vision backends uncopied](https://github.com/sgl-project/sglang/pull/35336) | `claude/serene-hopper-7bd733` | 52.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32147234710) |
| #32149182696<br>[#35341 [AMD][Fix] Qwen3.5: make empty-batch guard tuple-aware on fused AR+quant path](https://github.com/sgl-project/sglang/pull/35341) | `bingxche/fix-qwen35-fused-ar-tuple-guard` | 45.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32149182696) |
| #32115115505<br>[#27010 [HiCache] Fix PP inconsistency with HiCache L3 (#22607)](https://github.com/sgl-project/sglang/pull/27010) | `sglang_pp_bug4` | 44.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32115115505) |
| #32149737242<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `codex/diffusion-reuse-srt-qwen-vision` | 43.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32149737242) |
| #32152224544<br>[#28403 [PD] Introduce runtime role switching between prefill and decode](https://github.com/sgl-project/sglang/pull/28403) | `feat/pd-role-switch-mori-v2` | 38.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32152224544) |
| #32151944608<br>[#35343 Sync FlashInfer autotune tactic choice across TP ranks](https://github.com/sgl-project/sglang/pull/35343) | `mmangkad/flashinfer-autotune-sync-tactics-across-tp-ranks` | 23.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32151944608) |
| #32150110116<br>[#35342 [VLM] Route every multimodal processor through the worker pool's call site](https://github.com/sgl-project/sglang/pull/35342) | `claude/mm-processor-async-callsites` | 18.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32150110116) |
| #32114914873<br>[#24911 Profiling Enhancements [2/3]: detailed execution step annotations](https://github.com/sgl-project/sglang/pull/24911) | `main` | 15.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32114914873) |
| #32152788381<br>[#34713 [diffusion] Decouple encoder parallelism from the DiT parallel layout](https://github.com/sgl-project/sglang/pull/34713) | `encoder-dp-under-tp` | 8.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32152788381) |
| #32112220396<br>[#35286 [Fix] Assert the page-aligned SWA evict floor at PD decode prealloc](https://github.com/sgl-project/sglang/pull/35286) | `lsyin/pd-swa-evict-page-floor` | 6.2min | 8 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32112220396) |

---


## [Run #32152788381](https://github.com/sgl-project/sglang/actions/runs/32152788381)
- **分支**: `encoder-dp-under-tp`
- **总耗时**: 8.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32152788381

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 作业未显示实际测试失败，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32152788381/job/95762709857) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或提前结束，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32152788381/job/95762709857


## [Run #32152224544](https://github.com/sgl-project/sglang/actions/runs/32152224544)
- **分支**: `feat/pd-role-switch-mori-v2`
- **总耗时**: 38.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32152224544

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815384) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815472) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815688) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760816701) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760816712) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | PR健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760816751) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.2min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760816822) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node.js弃用警告及上传失败产物（无文件）等常规信息，未包含多模态生成测试的实际执行步骤或错误输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815384

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤后触发fast-fail机制，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815472

- **base-b-test-16-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败策略被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815688

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760816701

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760816712

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业在启动前的PR健康检查中发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业被跳过未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760816751

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760816822

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815507) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815526) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815553) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815701) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32152224544/job/95760815760) |


## [Run #32151944608](https://github.com/sgl-project/sglang/actions/runs/32151944608)
- **分支**: `mmangkad/flashinfer-autotune-sync-tactics-across-tp-ranks`
- **总耗时**: 23.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32151944608

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102503) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102633) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 环境问题 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102640) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102654) |
| base-b-test-8-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102740) |
| base-b-test-4-npu-a3 / run (0) | 1.4min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102902) |
| base-b-test-4-npu-a3 / run (1) | 1.1min | 其他 | 健康检查检测到其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102911) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102975) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现根因任务失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760103146) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760103168) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760103257) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | PR测试健康检查失败，根因是其他作业multimodal-gen-test-1-npu-a3失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760103387) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102503

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102633

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102640

- **base-b-test-2-npu-a3 / run (0)**: 本作业在启动前的PR健康检查中发现根因失败作业multimodal-gen-test-1-npu-a3，触发Fast-fail机制，主动跳过执行，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102654

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查识别出根因失败作业为multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102740

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102902

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3）因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102911

- **base-b-test-1-npu-a3 / run (0)**: 健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760102975

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败任务，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760103146

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段被健康检查拦截，检测到另一个作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业未实际执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760103168

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760103257

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 该作业在启动前的PR测试健康检查阶段被快速失败机制跳过，根因作业为multimodal-gen-test-1-npu-a3，本作业本身未执行实际测试，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95760103387

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32151944608/job/95759786803) |


## [Run #32150110116](https://github.com/sgl-project/sglang/actions/runs/32150110116)
- **分支**: `claude/mm-processor-async-callsites`
- **总耗时**: 18.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32150110116

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422390) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因任务失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422497) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422527) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422553) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422561) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422649) |
| base-b-test-4-npu-a3 / run (1) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422661) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业根因失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797423480) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.3min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797423602) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797423611) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.5min | 其他 | PR健康检查失败，根因是多模态生成测试失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797423624) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422390

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败任务，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422497

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被终止，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422527

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤，最终因快速失败策略终止，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422553

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3 / run (0)）因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422561

- **base-b-test-1-npu-a3 / run (0)**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤后快速失败，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422649

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因作业，因此本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422661

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797423480

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797423602

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797423611

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败机制跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797423624

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32150110116/job/95797422695) |


## [Run #32149737242](https://github.com/sgl-project/sglang/actions/runs/32149737242)
- **分支**: `codex/diffusion-reuse-srt-qwen-vision`
- **总耗时**: 43.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32149737242

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487443) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487543) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487545) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487553) |
| base-b-test-16-npu-a3 / run (0) | 2.3min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487566) |
| base-b-test-8-npu-a3 / run (0) | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487592) |
| base-b-test-4-npu-a3 / run (0) | 2.8min | 环境问题 | 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，导致级联跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487617) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487624) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 2.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487983) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752488028) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 2.7min | 其他 | 健康检查发现其他作业根因失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752488079) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | PR健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752488085) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487443

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487543

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487545

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤，实际未执行测试，因快速失败机制退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487553

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被终止，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487566

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487592

- **base-b-test-4-npu-a3 / run (0)**: 本作业在PR测试健康检查阶段被判定为级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制跳过后续测试，非本作业自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487617

- **base-b-test-4-npu-a3 / run (1)**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，被判定为根因，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487624

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752487983

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752488028

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752488079

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示本作业未实际运行，因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149737242/job/95752488085


## [Run #32149182696](https://github.com/sgl-project/sglang/actions/runs/32149182696)
- **分支**: `bingxche/fix-qwen35-fused-ar-tuple-guard`
- **总耗时**: 45.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32149182696

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 0.8min | 其他 | 健康检查级联失败导致作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751679712) |
| multimodal-gen-test-1-npu-a3 | 3.9min | 环境问题 | 作业因缺少失败产物文件而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751679764) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联取消 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751679813) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680006) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查失败，根因是多模态测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680034) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680061) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680067) |
| base-b-test-4-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680176) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680515) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680798) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680853) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751681563) |

- **base-a-test-1-npu-a2 / run (0)**: 该作业因其他根因作业（multimodal-gen-test-1-npu-a3）失败而被健康检查过滤，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751679712

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤未找到diffusion-failures/目录，说明测试未产生失败记录，作业可能因环境或前置条件未满足而终止，未进入核心测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751679764

- **base-b-test-8-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而触发快速失败机制，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751679813

- **base-b-test-4-npu-a3 / run (1)**: 该作业本身未执行测试，因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，被级联跳过并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680006

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-16-npu-a3）因级联失败被过滤，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680034

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680061

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤，实际未执行测试，属于上游失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680067

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680176

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段因健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680515

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680798

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751680853

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示该作业在健康检查阶段被识别为级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32149182696/job/95751681563


## [Run #32147234710](https://github.com/sgl-project/sglang/actions/runs/32147234710)
- **分支**: `claude/serene-hopper-7bd733`
- **总耗时**: 52.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32147234710

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744902894) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他作业根因失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903227) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903424) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903468) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903476) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903522) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是多模态测试作业失败导致级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903631) |
| base-b-test-16-npu-a3 / run (0) | 2.3min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903794) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744904398) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744904425) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业根因失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744904644) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | PR测试健康检查失败，根因是多模态测试作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744904714) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node.js弃用警告及上传diffusion-failures工件时未找到文件，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744902894

- **base-b-test-2-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903227

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903424

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903468

- **base-b-test-8-npu-a3 / run (0)**: 作业在“Check PR test health”步骤中检测到multimodal-gen-test-1-npu-a3为根因失败，按策略跳过本作业，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903476

- **base-b-test-4-npu-a3 / run (1)**: 作业启动后健康检查发现multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903522

- **base-b-test-4-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903631

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744903794

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744904398

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744904425

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败后，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744904644

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147234710/job/95744904714


## [Run #32147191579](https://github.com/sgl-project/sglang/actions/runs/32147191579)
- **分支**: `nemotron-3.5-spec-comparison`
- **总耗时**: 53.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32147191579

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 环境问题 | Git 拉取代码失败，远端仓库缺少指定 commit。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743927675) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928077) |
| base-b-test-8-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928163) |
| base-b-test-1-npu-a3 / run (0) | 1.4min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928185) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现多个根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928234) |
| base-b-test-4-npu-a3 / run (0) | 1.4min | 其他 | 健康检查快速失败机制触发，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928313) |
| base-b-test-2-npu-a3 / run (0) | 1.4min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928353) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928382) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.2min | 环境问题 | Git 拉取 PR 合并提交失败，远端仓库不存在该 ref。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743929615) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | 环境问题 | GitHub Actions 无法获取 PR 合并后的 commit，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743929627) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.5min | 环境问题 | GitHub Actions 无法从远程仓库获取指定 PR 合并提交，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743929690) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.1min | 环境问题 | Git 拉取合并提交时找不到指定 ref，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743929831) |

- **multimodal-gen-test-1-npu-a3**: 作业在 checkout 阶段执行 git fetch 时，远端返回 'not our ref'，多次重试均失败，导致无法获取 PR 合并提交，属于仓库状态或缓存问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743927675

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查检测到多个根因失败作业（如multimodal-gen-test-1-npu-a3等），本作业因级联失败被过滤后快速失败，并非自身代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928077

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3、base-c-test-acc-2-npu-a3等根因作业失败，本作业作为级联失败被过滤，最终因快速失败策略退出，非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928163

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-2-npu-a3两个根因作业失败，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928185

- **base-b-test-16-npu-a3 / run (0)**: 健康检查过滤级联失败后，识别出multimodal-gen-test-1-npu-a3等4个根因作业失败，触发fast-fail跳过本作业，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928234

- **base-b-test-4-npu-a3 / run (0)**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3及base-c-test-acc系列，本作业因健康检查过滤被标记为级联失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928313

- **base-b-test-2-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-2-npu-a3两个根因作业失败，触发了fast-fail机制，本作业未实际运行即被终止，属于依赖的上游失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928353

- **base-b-test-4-npu-a3 / run (1)**: 健康检查检测到multimodal-gen-test-1-npu-a3等4个根因作业失败，本作业被判定为级联失败而快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743928382

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: checkout 时 fetch 提交 0f9d407 失败，报错 'not our ref'，重试三次均失败，可能是 PR 已更新或缓存未同步，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743929615

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 git fetch 时远程仓库报错 'not our ref 0f9d407...'，重试三次均失败。可能是 PR 已更新或缓存不一致，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743929627

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 git fetch 时远程返回 'not our ref 0f9d407...'，即该 PR 的合并提交在远程仓库中不存在或已过期，多次重试均失败，最终作业退出。这属于 CI 基础设施或 PR 状态问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743929690

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业尝试从远程仓库拉取 PR 合并提交 0f9d407，但 git-cdn 服务返回 "not our ref"，多次重试均失败，最终退出码 128，属于基础设施或缓存同步问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32147191579/job/95743929831


## [Run #32146951488](https://github.com/sgl-project/sglang/actions/runs/32146951488)
- **分支**: `claude/paddleocr-support-optimization-bfaf52`
- **总耗时**: 57.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32146951488

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.9min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810185) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810612) |
| base-b-test-4-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810677) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810739) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810743) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810801) |
| base-b-test-16-npu-a3 / run (0) | 3.5min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810853) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他作业根因失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810869) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744811779) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744811968) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744812031) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.4min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744812230) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，随后清理退出。未出现测试执行、失败断言或超时信息，可能因前置条件未满足导致测试被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810185

- **base-b-test-8-npu-a3 / run (0)**: 健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810612

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810677

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，导致本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810739

- **base-a-test-1-npu-a2 / run (0)**: 本作业在健康检查阶段发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail跳过执行，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810743

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810801

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810853

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3为根因失败作业，本作业因快速失败（fast-fail）被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744810869

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744811779

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744811968

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744812031

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146951488/job/95744812230


## [Run #32146312799](https://github.com/sgl-project/sglang/actions/runs/32146312799)
- **分支**: `fix_ring_attention_npu`
- **总耗时**: 57.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32146312799

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 | 8.2min | 环境问题 | GitHub Actions 下载 action 时网络超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740946703) |
| multimodal-gen-test-1-npu-a3 | 9.8min | 环境问题 | GitHub Actions 下载 action 时网络超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740946867) |
| base-a-test-1-npu-a2 / run (0) | 2.1min | 其他 | 健康检查快速失败，根因是多模态测试作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740946922) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947026) |
| base-b-test-2-npu-a3 / run (0) | 1.3min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947053) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947104) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947128) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947152) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 级联失败，根因是其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947153) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.2min | 其他 | 作业因健康检查快速失败机制被跳过，根因是其他作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947493) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947534) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.1min | 其他 | 作业因其他根因任务失败被快速失败机制跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947547) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947681) |

- **multimodal-gen-test-2-npu-a3**: 日志显示在准备 action 时，HTTP 请求超时（00:01:40），重试后成功下载，但整体作业因网络不稳定而中断，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740946703

- **multimodal-gen-test-1-npu-a3**: 日志显示在准备 action 时，HTTP 请求超时（00:01:40），重试后成功下载了 checkout 和 upload-artifact，但后续作业未正常执行，最终无测试产物上传，属于网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740946867

- **base-a-test-1-npu-a2 / run (0)**: 该作业因PR健康检查检测到根因作业multimodal-gen-test-1/2-npu-a3失败，触发fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740946922

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3为根因失败，本作业因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947026

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，发现同批次其他作业（multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3）已失败，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947053

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤掉级联失败后，根因失败作业为multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947104

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3为根因失败作业，本作业因级联失败被过滤，随后触发fast-fail跳过执行，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947128

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查发现multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3两个根因作业失败，触发fast-fail机制，本作业被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947152

- **base-b-test-4-npu-a3 / run (0)**: 日志显示本作业因健康检查过滤了级联失败，根因作业为multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3，本作业并非直接失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947153

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业在启动阶段被健康检查过滤，判定为级联失败，实际根因是多模态生成测试（multimodal-gen-test-1/2-npu-a3）失败，本作业未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947493

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3两个根因作业失败，本作业因快速失败策略被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947534

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示该作业在健康检查阶段被过滤为级联失败，根因是multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3失败，导致本作业被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947547

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-2-npu-a3和multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32146312799/job/95740947681


## [Run #32144736340](https://github.com/sgl-project/sglang/actions/runs/32144736340)
- **分支**: `fix/xgrammar-mistral-common`
- **总耗时**: 104.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32144736340

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 33.2min | 其他 | 作业未显示实际测试失败，仅上传diffusion-failures目录时未找到文件，可能测试未运行或全部通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735769207) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95745545879) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95753882382) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95760445813) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95770236131) |

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤因diffusion-failures目录不存在而跳过上传，未出现测试失败或错误信息，可能测试未执行或结果正常，需进一步检查完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735769207

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段因健康检查检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95745545879

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95753882382

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95760445813

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95770236131

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735769591) |
| base-b-test-16-npu-a3 / run (0) | 54.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735769750) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735769848) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735769862) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735769895) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735769942) |
| base-b-test-2-npu-a3 / run (0) | 17.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735770078) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735771914) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735772200) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735772243) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144736340/job/95735772410) |


## [Run #32144619391](https://github.com/sgl-project/sglang/actions/runs/32144619391)
- **分支**: `feature/unified-runtime-attach`
- **总耗时**: 62.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32144619391

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070629) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070650) |
| base-b-test-2-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070723) |
| base-b-test-1-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070756) |
| base-b-test-8-npu-a3 / run (0) | 0.9min | 其他 | 健康检查快速失败，根因是多模态测试失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070820) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070834) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070839) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737071315) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 该作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737071394) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737071424) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737071435) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志或测试输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070629

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070650

- **base-b-test-2-npu-a3 / run (0)**: 该作业在启动前的健康检查中检测到根因作业multimodal-gen-test-1-npu-a3失败，因此被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070723

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070756

- **base-b-test-8-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070820

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070834

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070839

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败机制跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737071315

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业被级联跳过，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737071394

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业因PR健康检查机制被跳过，根因是多模态生成测试（multimodal-gen-test-1-npu-a3）失败，本作业被级联过滤，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737071424

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业后，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737071435

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 53.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32144619391/job/95737070671) |


## [Run #32143584692](https://github.com/sgl-project/sglang/actions/runs/32143584692)
- **分支**: `dsv4_state_pool_size`
- **总耗时**: 138.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32143584692

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.0min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978062) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 21.1min | 性能回归 | NPU性能测试中deepseek_v4_flash用例失败，未达性能目标 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95753437151) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 35.5min | 性能回归 | NPU性能测试中qwen3_6_27b用例失败，未达性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95766065443) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含实际测试命令或失败断言，仅显示runner初始化、Node版本警告及上传diffusion-failures目录（无文件）。可能测试未运行或日志被截断，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978062

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms用例退出码1，耗时336秒，未满足50ms性能要求，疑似性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95753437151

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试套件中qwen3_6_27b_w8a8_1p_in64k_out1k_50ms用例退出码1，其余用例通过，判定为性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95766065443

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978253) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978300) |
| base-b-test-16-npu-a3 / run (0) | 75.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978310) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978333) |
| base-b-test-2-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978381) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978403) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978420) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978690) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 47.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978708) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978767) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95731978851) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 35.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95739426160) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143584692/job/95749962372) |


## [Run #32143251403](https://github.com/sgl-project/sglang/actions/runs/32143251403)
- **分支**: `khoa/fix-required-tool-choice-json-fallback`
- **总耗时**: 122.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32143251403

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.7min | 环境问题 | 作业因缺少失败产物文件而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964347) |
| base-a-test-1-npu-a2 / run (0) | 1.0min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964696) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964888) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735965293) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735965329) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735965363) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.1min | 精度回归 | NPU精度测试中qwen3_5_9b用例失败，导致整体测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735965538) |

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures目录时提示无文件，说明测试未产生失败样本，作业可能因环境配置或前置步骤异常而中断，未进入核心测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964347

- **base-a-test-1-npu-a2 / run (0)**: 健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964696

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-4-npu-a3）被Fast-fail机制跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964888

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735965293

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735965329

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 本作业在启动前的PR健康检查中，检测到根因作业multimodal-gen-test-1-npu-a3失败，触发了fast-fail机制，本作业未实际执行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735965363

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件base-c-test-acc-2-npu-a3中，moonshotai_moonlight_16b_a3b用例通过，但qwen3_5_9b_bf16_1p_gsm8k用例退出码为1，耗时3835秒，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735965538

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964717) |
| base-b-test-16-npu-a3 / run (0) | 55.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964737) |
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964746) |
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964759) |
| base-b-test-8-npu-a3 / run (0) | 9.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32143251403/job/95735964849) |


## [Run #32142106893](https://github.com/sgl-project/sglang/actions/runs/32142106893)
- **分支**: `amd/dsv4-shared-experts-fusion-top6`
- **总耗时**: 102.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32142106893

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122189) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122387) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 68.1min | 精度回归 | Qwen3.5-9B GSM8K 精度测试失败，0/3 用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122924) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 3.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95737095498) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.3min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95743337117) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败（fast-fail）跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95757849786) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本或提前退出，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122189

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，检测到同一次PR运行中multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122387

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 test_npu_qwen3_5_9b_bf16_1p_gsm8k.py 运行 3886 秒后退出码为 1，所有 3 个精度用例均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122924

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发快速失败机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95737095498

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95743337117

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-acc-2-npu-a3，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95757849786

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122285) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122317) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122336) |
| base-b-test-1-npu-a3 / run (0) | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122397) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122427) |
| base-b-test-16-npu-a3 / run (0) | 52.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727122430) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 70.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727123020) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727123026) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 34.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32142106893/job/95727123200) |


## [Run #32138292194](https://github.com/sgl-project/sglang/actions/runs/32138292194)
- **分支**: `patch-8`
- **总耗时**: 135.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32138292194

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.9min | 环境问题 | GitHub Actions 下载 upload-artifact 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166556) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 36.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95735532655) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.6min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95745556312) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95757829621) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，虽然后续重试成功，但可能影响作业稳定性。此外，Node 20 弃用警告和 diffusion-failures 目录无文件上传，均非根本原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166556

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95735532655

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动前的健康检查中检测到同一PR的另一个作业base-c-test-perf-16-npu-a3失败，被判定为根因失败，因此本作业被快速失败跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95745556312

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业在启动前的PR健康检查阶段被快速失败（fast-fail），原因是同一次运行中另一个作业base-c-test-perf-16-npu-a3失败被判定为根因，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95757829621

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166685) |
| base-b-test-16-npu-a3 / run (0) | 63.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166768) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166789) |
| base-b-test-1-npu-a3 / run (0) | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166807) |
| base-b-test-4-npu-a3 / run (1) | 35.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166817) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166898) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717166911) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 55.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717167143) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717167193) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717167243) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 33.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95717167286) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32138292194/job/95724991388) |


## [Run #32132517558](https://github.com/sgl-project/sglang/actions/runs/32132517558)
- **分支**: `claude/paddleocr-support-optimization-bfaf52`
- **总耗时**: 92.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32132517558

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 17.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715611982) |
| base-b-test-4-npu-a3 / run (1) | 1.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715611985) |
| base-b-test-16-npu-a3 / run (0) | 1.9min | 环境问题 | GitHub API 返回 500 错误导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715611988) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 80.7min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612573) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.4min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612619) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95724780158) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.2min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95728633699) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（未找到文件），未出现任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715611982

- **base-b-test-4-npu-a3 / run (1)**: 作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，属于根因失败，因此本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715611985

- **base-b-test-16-npu-a3 / run (0)**: github-script 调用 GitHub API 查询 lint check-runs 时收到 500 服务器错误，属于 GitHub 服务端临时故障，非代码或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715611988

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612573

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612619

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因作业，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95724780158

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95728633699

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715611938) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612061) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612136) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612179) |
| base-b-test-4-npu-a3 / run (0) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612218) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612520) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32132517558/job/95715612585) |


## [Run #32131837084](https://github.com/sgl-project/sglang/actions/runs/32131837084)
- **分支**: `pllimax/output-log-dir-structure`
- **总耗时**: 251.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32131837084

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 141.6min | 超时 | 性能测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95723698214) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 68.6min | 性能回归 | NPU性能测试中w8a8长序列用例未达预期，退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95741330279) |

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试 test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py 在启动服务器后运行超过7800秒未完成，被强制终止，最终0/4测试通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95723698214

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py失败，耗时676秒，可能因性能未达50ms目标或运行错误，需检查日志确认具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95741330279

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697507693) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697507760) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697507775) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697507805) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697507823) |
| base-b-test-16-npu-a3 / run (0) | 51.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697507855) |
| base-b-test-4-npu-a3 / run (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697507976) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 41.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697508193) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697508203) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 97.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697508275) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95697508306) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95716047129) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 34.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32131837084/job/95722379398) |


## [Run #32121587014](https://github.com/sgl-project/sglang/actions/runs/32121587014)
- **分支**: `fix/kimi-k3-deferred-grids-global-index`
- **总耗时**: 115.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32121587014

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境警告和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788194923) |
| base-b-test-4-npu-a3 / run (1) | 1.4min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195092) |
| base-b-test-8-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195124) |
| base-b-test-4-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195149) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | PR健康检查失败，因其他作业失败触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95790821262) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业根因失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95800584944) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95801913870) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 作业因健康检查检测到其他根因作业失败而被快速跳过，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95820150840) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示Node.js 20弃用警告和diffusion-failures目录无文件上传，未发现明确错误信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788194923

- **base-b-test-4-npu-a3 / run (1)**: 健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业被标记为级联失败并快速跳过，非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195092

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，本作业因级联失败被过滤，最终因根因作业失败而快速失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195124

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，属于根因失败，因此本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195149

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail跳过本作业，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95790821262

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95800584944

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行PR测试健康检查时，检测到multimodal-gen-test-1-npu-a3作业失败（根因），触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95801913870

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业被级联过滤后触发fast-fail机制，属于级联失败，非本作业代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95820150840

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788194995) |
| base-a-test-1-npu-a2 / run (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195150) |
| base-b-test-16-npu-a3 / run (0) | 45.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195165) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195206) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 103.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195512) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195666) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 48.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195703) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32121587014/job/95788195707) |


## [Run #32117587776](https://github.com/sgl-project/sglang/actions/runs/32117587776)
- **分支**: `perf/dp-global-prefill-interval`
- **总耗时**: 386.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32117587776

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.0min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593117) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 72.3min | 性能回归 | NPU性能测试用例失败，w8a8长序列测试未达性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95685253776) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593117

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py退出码1，该用例为性能测试，可能因推理速度未达50ms阈值而失败，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95685253776

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593239) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593251) |
| base-b-test-16-npu-a3 / run (0) | 52.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593321) |
| base-b-test-4-npu-a3 / run (1) | 10.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593326) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593330) |
| base-b-test-4-npu-a3 / run (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593339) |
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593414) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593879) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593880) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 45.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593912) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95650593930) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 26.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95663829816) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95672567662) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 272.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32117587776/job/95677044654) |


## [Run #32115115505](https://github.com/sgl-project/sglang/actions/runs/32115115505)
- **分支**: `sglang_pp_bug4`
- **总耗时**: 44.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32115115505

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890188) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890474) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890493) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890507) |
| base-b-test-16-npu-a3 / run (0) | 2.0min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890532) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890600) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890629) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.6min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642891138) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败而被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642891242) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642891274) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.5min | 环境问题 | GitHub API 请求失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95649304381) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node.js版本弃用警告和上传artifact时无文件提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890188

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890474

- **base-b-test-1-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3，本作业作为级联失败被过滤后触发fast-fail机制，提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890493

- **base-b-test-2-npu-a3 / run (0)**: 本作业在“Check PR test health”步骤中，检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3为根因失败，因此触发fast-fail机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890507

- **base-b-test-16-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3为根因失败作业，本作业因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890532

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是多模态测试和base-c测试失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890600

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3，本作业因快速失败机制被跳过，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890629

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 在安装 Rust 工具链时，从内部缓存服务下载 rust-1.92 的 channel 文件超时，导致脚本退出，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642891138

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3，本作业被级联过滤后快速失败，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642891242

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3失败，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642891274

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: github-script 步骤调用 GitHub API 查询 lint check-runs 时返回 500 错误，可能是 GitHub 服务临时故障或网络问题，导致作业在测试开始前即失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95649304381

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642890611) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32115115505/job/95642891140) |


## [Run #32114914873](https://github.com/sgl-project/sglang/actions/runs/32114914873)
- **分支**: `main`
- **总耗时**: 15.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32114914873

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.2min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258122) |
| base-b-test-2-npu-a3 / run (0) | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258187) |
| base-b-test-4-npu-a3 / run (0) | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258201) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258240) |
| base-b-test-1-npu-a3 / run (0) | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258266) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258324) |
| base-b-test-8-npu-a3 / run (0) | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258325) |
| base-b-test-16-npu-a3 / run (0) | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258348) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258535) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258754) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258765) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258776) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258122

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258187

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258201

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258240

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258266

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258324

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258325

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258348

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258535

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258754

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258765

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32114914873/job/95642258776


## [Run #32112220396](https://github.com/sgl-project/sglang/actions/runs/32112220396)
- **分支**: `lsyin/pd-swa-evict-page-floor`
- **总耗时**: 6.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32112220396

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694667) |
| base-b-test-2-npu-a3 / run (0) | 2.4min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，且拉取 PR 合并分支 refs/pull/35286/merge 失败，导致作业无法开始。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694756) |
| base-a-test-1-npu-a2 / run (0) | 1.5min | 环境问题 | GitHub Actions 无法获取 PR 的 merge ref，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694794) |
| base-b-test-8-npu-a3 / run (0) | 1.7min | 环境问题 | GitHub Actions 无法获取 PR 合并分支 refs/pull/35286/merge，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694857) |
| base-b-test-16-npu-a3 / run (0) | 1.8min | 环境问题 | GitHub Actions 无法获取 PR 的 merge 引用，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694879) |
| base-b-test-4-npu-a3 / run (0) | 1.8min | 环境问题 | GitHub Actions 无法获取 PR 的 merge ref，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694918) |
| base-b-test-4-npu-a3 / run (1) | 2.1min | 环境问题 | GitHub Actions 无法获取 PR 合并引用，导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694977) |
| base-b-test-1-npu-a3 / run (0) | 1.9min | 环境问题 | GitHub Actions 无法获取 PR 合并引用，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695009) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695295) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695351) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.7min | 环境问题 | PR健康检查发现多个根因作业失败，触发快速失败机制，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695462) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695559) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未显示任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694667

- **base-b-test-2-npu-a3 / run (0)**: 日志显示下载 actions/checkout 时返回 429 Too Many Requests，重试后仍失败；随后 git fetch 找不到 refs/pull/35286/merge，可能 PR 已关闭或不存在，最终退出码 128。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694756

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 git fetch 多次尝试拉取 refs/pull/35286/merge 均失败，报错 'couldn't find remote ref'，可能是 PR 已关闭、合并或仓库状态异常，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694794

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 git fetch 多次重试均报错 "couldn't find remote ref refs/pull/35286/merge"，可能是 PR 已关闭、分支被删除或仓库缓存未同步，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694857

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 git fetch 时找不到 refs/pull/35286/merge，重试三次均失败。可能是 PR 已关闭、分支被删除或仓库缓存问题，属于环境或仓库状态异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694879

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 git fetch 多次尝试拉取 refs/pull/35286/merge 均失败，提示 couldn't find remote ref，可能是 PR 已关闭、合并或仓库缓存问题，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694918

- **base-b-test-4-npu-a3 / run (1)**: 作业在 checkout 阶段尝试获取 refs/pull/35286/merge 引用，但远程仓库中不存在该引用，重试三次均失败。可能是 PR 已关闭、合并分支被删除或 CDN 缓存问题，属于环境或仓库状态异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859694977

- **base-b-test-1-npu-a3 / run (0)**: 作业在 fetch refs/pull/35286/merge 时多次失败，提示找不到该远程引用，可能是 PR 已关闭、分支被删除或仓库缓存问题，最终导致作业退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695009

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示health-check检测到base-a-test-1-npu-a2和base-b-test-8-npu-a3两个根因作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695295

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到base-a-test-1-npu-a2等三个根因作业失败，本作业被级联跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695351

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查检测到base-a-test-1-npu-a2等4个根因作业失败，本作业作为级联失败被过滤并跳过，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695462

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到base-a-test-1-npu-a2等3个根因作业失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112220396/job/95859695559


---
*Auto-generated by npu_pr_monitor.py*