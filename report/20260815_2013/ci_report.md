# NPU CI 执行监控
**生成时间**: 2026-08-15 12:13 UTC
**分析 Run 数**: 27

---

## 📊 本次执行总结

- **成功 Job 数**: 186
- **失败 Run 数**: 21
- **成功 Job 平均耗时**: 25.0min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 263.1min | #31868231108 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94975137035) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 128.1min | #31869972200 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022547) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 123.8min | #31868247127 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611606) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.7min | #31866988995 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969594109) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.5min | #31862528793 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297077) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.2min | #31868231108 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579851) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.5min | #31870165803 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529301) |
| base-b-test-16-npu-a3 / run (0) | 80.3min | #31860134573 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975670) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 77.5min | #31868231108 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94981531510) |
| base-b-test-16-npu-a3 / run (0) | 56.3min | #31862528793 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297001) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31868247127<br>[#33604 Fix Whisper transcription for audio over 30 seconds](https://github.com/sgl-project/sglang/pull/33604) | `agent/whisper-long-audio-chunking` | 143.7min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31868247127) |
| #31869972200<br>[#34715 [bugfix] [NPU] fix transpose batch matmul K*B exceed 65536.](https://github.com/sgl-project/sglang/pull/34715) | `bmm65536-fallback` | 135.2min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31869972200) |
| #31862099662<br>[#34668 fix(ci): refresh nightly precision baseline from remote](https://github.com/sgl-project/sglang/pull/34668) | `xinyuan/nightly-precision-stale-baseline` | 132.1min | 2 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31862099662) |
| #31870692407<br>[#32597 Support streaming session on NPU](https://github.com/sgl-project/sglang/pull/32597) | `streaming_session` | 128.7min | 4 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31870692407) |
| #31866988995<br>[#34509 [JIT Kernel] Migrate moe_topk_softmax from AOT to JIT](https://github.com/sgl-project/sglang/pull/34509) | `voidc-minor/jit-moe-topk-softmax` | 109.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31866988995) |
| #31860134573<br>[#34284 fix(scheduler): track max prefill batch size over recent real admissions](https://github.com/sgl-project/sglang/pull/34284) | `main` | 107.7min | 6 | base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31860134573) |
| #31870165803<br>[#33726 fix(bcg): preserve Qwen3-VL DeepStack inputs during replay](https://github.com/sgl-project/sglang/pull/33726) | `fix/bcg-deepstack-replay-slot` | 98.7min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31870165803) |
| #31862528793<br>[#34558 [Bugfix] Preserve MXFP4 Triton weights in sharded state](https://github.com/sgl-project/sglang/pull/34558) | `fix-mxfp4-sharded-state` | 93.3min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31862528793) |
| #31859834635<br>[#30805 [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103](https://github.com/sgl-project/sglang/pull/30805) | `dsv4_fp8_trtllm_gen` | 79.2min | 4 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31859834635) |
| #31860091936<br>[#34880 fix: honor explicit model loader classes](https://github.com/sgl-project/sglang/pull/34880) | `codex/honor-explicit-model-loader` | 52.9min | 5 | base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31860091936) |
| #31861466139<br>[#32597 Support streaming session on NPU](https://github.com/sgl-project/sglang/pull/32597) | `streaming_session` | 48.9min | 7 | base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-b-test-2-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31861466139) |
| #31869685222<br>[#34913 [CI] Move the static ratchets back to CPU unit tests](https://github.com/sgl-project/sglang/pull/34913) | `main` | 44.3min | 7 | base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31869685222) |
| #31862741908<br>[#33726 fix(bcg): preserve Qwen3-VL DeepStack inputs during replay](https://github.com/sgl-project/sglang/pull/33726) | `fix/bcg-deepstack-replay-slot` | 42.0min | 6 | base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31862741908) |
| #31868373873<br>[#34880 fix: honor explicit model loader classes](https://github.com/sgl-project/sglang/pull/34880) | `main` | 31.8min | 9 | base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31868373873) |
| #31869051017<br>[#33726 fix(bcg): preserve Qwen3-VL DeepStack inputs during replay](https://github.com/sgl-project/sglang/pull/33726) | `fix/bcg-deepstack-replay-slot` | 27.4min | 7 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31869051017) |
| #31868629427<br>[#34913 [CI] Move the static ratchets back to CPU unit tests](https://github.com/sgl-project/sglang/pull/34913) | `lsyin/remove-static-ratchets` | 25.7min | 8 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31868629427) |
| #31864801870<br>[#34769 [AMD][CI] Fix stage-b: AttributeError on multimodal embedding requests](https://github.com/sgl-project/sglang/pull/34769) | `main` | 22.2min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31864801870) |
| #31866803987<br>[#32593 [Kernel] Enable Helion backend for Kimi Delta-Attention](https://github.com/sgl-project/sglang/pull/32593) | `main` | 15.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31866803987) |
| #31866229885<br>[#34789 [MoE] Route every trtllm-gen MoE call site through one PDL guard](https://github.com/sgl-project/sglang/pull/34789) | `main` | 14.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31866229885) |
| #31865143921<br>[#33685 [NPU CI] Reorganize test output/log directory structure with workflow context](https://github.com/sgl-project/sglang/pull/33685) | `pllimax/output-log-dir-structure` | 13.4min | 4 | base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31865143921) |
| #31865824658<br>[#29328 [AMD][Quantization] Online MXFP4 quantization 4/N - NVFP4 to MXFP4 Online Requantization on AMD GPUs](https://github.com/sgl-project/sglang/pull/29328) | `main` | 9.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31865824658) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 22 | 22 | 0 | 0 |
| base-b-test-1-npu-a3 / run (0) | 22 | 13 | 0 | 9 |
| base-b-test-16-npu-a3 / run (0) | 22 | 12 | 1 | 9 |
| base-b-test-2-npu-a3 / run (0) | 22 | 15 | 0 | 7 |
| base-b-test-4-npu-a3 / run (0) | 22 | 12 | 0 | 10 |
| base-b-test-4-npu-a3 / run (1) | 22 | 17 | 0 | 5 |
| base-b-test-8-npu-a3 / run (0) | 22 | 19 | 0 | 3 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22 | 14 | 1 | 7 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 22 | 6 | 0 | 16 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 22 | 13 | 0 | 9 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 22 | 19 | 0 | 3 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 14 | 1 | 0 | 13 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 6 | 1 | 0 | 5 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 13 | 2 | 0 | 11 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 19 | 2 | 0 | 17 |
| multimodal-gen-test-1-npu-a3 | 24 | 18 | 0 | 6 |

---


## [Run #31870692407](https://github.com/sgl-project/sglang/actions/runs/31870692407)
- **分支**: `streaming_session`
- **总耗时**: 128.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31870692407

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 126.5min | 精度回归 | qwen3_5_9b 精度测试失败，导致作业整体退出码非零。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788279) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.7min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94979253524) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.4min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94981897584) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94984802329) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k.py 测试失败（exit code 1），其余两个测试通过。该测试属于精度回归测试，可能因模型精度不达标或代码改动导致结果偏差。
  链接: https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788279

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1034秒后失败，该测试为性能测试，0/1通过，表明性能指标未达到预期要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94979253524

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因作业，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94981897584

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3作业失败，作为根因作业触发了快速失败，导致本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94984802329

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 51.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788103) |
| base-a-test-1-npu-a2 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788107) |
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788115) |
| base-b-test-2-npu-a3 / run (0) | 19.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788134) |
| base-b-test-4-npu-a3 / run (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788143) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788161) |
| multimodal-gen-test-1-npu-a3 | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788176) |
| base-b-test-4-npu-a3 / run (1) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788228) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788305) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788391) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870692407/job/94978788422) |


## [Run #31870165803](https://github.com/sgl-project/sglang/actions/runs/31870165803)
- **分支**: `fix/bcg-deepstack-replay-slot`
- **总耗时**: 98.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31870165803

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94978011998) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94980363881) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94982139846) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他性能作业失败，导致本作业被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94986208686) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，该测试为性能测试，预期耗时约3600秒，实际运行1125秒后退出，可能因性能指标未达预期或运行异常导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94978011998

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94980363881

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，本作业作为级联失败被过滤，最终因根因作业失败而快速失败，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94982139846

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3等作业失败，本作业作为级联失败被跳过，实际未执行测试，属于上游失败引发的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94986208686

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529095) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529096) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529107) |
| base-b-test-8-npu-a3 / run (0) | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529111) |
| multimodal-gen-test-1-npu-a3 | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529116) |
| base-b-test-4-npu-a3 / run (0) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529120) |
| base-b-test-2-npu-a3 / run (0) | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529135) |
| base-b-test-16-npu-a3 / run (0) | 47.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529160) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529256) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529267) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529301) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31870165803/job/94977529325) |


## [Run #31869972200](https://github.com/sgl-project/sglang/actions/runs/31869972200)
- **分支**: `bmm65536-fallback`
- **总耗时**: 135.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31869972200

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94978245855) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现同批次其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94980664106) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94983096330) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94990193338) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1138秒后失败，该测试为性能基准测试，要求50ms延迟，实际未达标，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94978245855

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因作业，导致本作业（base-c-test-perf-16-npu-a3）在启动前被快速失败跳过，并非本作业自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94980664106

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（4-npu）因级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94983096330

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94990193338

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022293) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022391) |
| base-b-test-8-npu-a3 / run (0) | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022395) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022423) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022437) |
| base-b-test-4-npu-a3 / run (0) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022451) |
| base-b-test-16-npu-a3 / run (0) | 53.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022460) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022469) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022525) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 128.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022547) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022583) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869972200/job/94977022604) |


## [Run #31869766261](https://github.com/sgl-project/sglang/actions/runs/31869766261)
- **分支**: `agent/minimax-h3-b300-high-quality`
- **总耗时**: 42.1min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31869766261

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869766261/job/94976509158) |


## [Run #31869685222](https://github.com/sgl-project/sglang/actions/runs/31869685222)
- **分支**: `main`
- **总耗时**: 44.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31869685222

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 1.2min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346645) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 代码错误 | HiCache MLA 测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346658) |
| base-b-test-1-npu-a3 / run (0) | 41.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346660) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 43.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346853) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94977051348) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.1min | 其他 | 作业因其他根因作业失败被快速失败跳过，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94979060707) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94980329253) |

- **base-b-test-16-npu-a3 / run (0)**: 该作业在启动阶段被健康检查脚本判定为级联失败，根因是另一个作业base-b-test-4-npu-a3失败，本作业被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346645

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试文件在运行约292秒后失败，退出码为1，导致整个作业失败。具体失败原因需查看该测试文件的详细输出，可能是测试断言失败或运行时错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346658

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常，但在07:15:57出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346660

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346853

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到 base-b-test-4-npu-a3 作业失败，作为根因作业触发了 fast-fail 机制，导致本作业未实际运行即被终止，属于上游失败引发的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94977051348

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现根因作业base-b-test-4-npu-a3失败，本作业被级联跳过，日志显示Fast-fail，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94979060707

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因作业 base-b-test-4-npu-a3 / run (0) 失败，本作业作为级联失败被过滤，最终因快速失败策略被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94980329253

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346564) |
| base-b-test-2-npu-a3 / run (0) | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346639) |
| base-b-test-4-npu-a3 / run (1) | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346654) |
| base-b-test-8-npu-a3 / run (0) | 10.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346683) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346723) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346768) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346794) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869685222/job/94976346847) |


## [Run #31869352179](https://github.com/sgl-project/sglang/actions/runs/31869352179)
- **分支**: `agent/fix-diffusion-weight-lock-filename`
- **总耗时**: 52.2min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31869352179

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869352179/job/94975484755) |


## [Run #31869051017](https://github.com/sgl-project/sglang/actions/runs/31869051017)
- **分支**: `fix/bcg-deepstack-replay-slot`
- **总耗时**: 27.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31869051017

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974710977) |
| base-b-test-4-npu-a3 / run (0) | 26.5min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974710983) |
| base-b-test-16-npu-a3 / run (0) | 24.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711035) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 16.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711064) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711082) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8.7min | 环境问题 | 自定义容器执行失败，NPU图捕获过程中容器异常退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94975463277) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94977162306) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974710977

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试用例TestNpuSpeculativeTokenMap.test_eagle_with_valid_token_map_gsm8k启动后，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974710983

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载至38%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711035

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但执行自定义容器时失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711064

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常，但随后报错“Executing the custom container implementation failed”，提示联系runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711082

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在NPU图捕获（Capturing batches）进行到20%时，自定义容器实现执行失败，导致作业终止。这属于自托管runner环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94975463277

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行自定义容器实现时失败，错误为"Executing the custom container implementation failed"，提示联系runner管理员，属于基础设施环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94977162306

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974710974) |
| base-b-test-2-npu-a3 / run (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974710978) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974710995) |
| base-a-test-1-npu-a2 / run (0) | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711004) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711016) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711061) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31869051017/job/94974711063) |


## [Run #31868629427](https://github.com/sgl-project/sglang/actions/runs/31868629427)
- **分支**: `lsyin/remove-static-ratchets`
- **总耗时**: 25.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31868629427

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617464) |
| base-b-test-16-npu-a3 / run (0) | 1.2min | 环境问题 | Kubernetes Pod 启动失败，导致作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617582) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617599) |
| base-b-test-4-npu-a3 / run (0) | 24.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617625) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 23.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617733) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617777) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617781) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 12.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94974393459) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifacts（无文件）等常规信息，未出现测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617464

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 Pod linux-aarch64-a3-16-cn12-001-772vk-runner-b6l4s-workflow 处于 Failed 状态，无法上线，可能是资源不足、镜像拉取失败或节点故障，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617582

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试进行到第9/10个文件时，runner报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617599

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617625

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在06:32:12出现错误'Executing the custom container implementation failed'，提示联系self-hosted runner管理员，属于runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617733

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在运行约25分钟后，日志显示"Executing the custom container implementation failed"，提示联系runner管理员，属于NPU自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617777

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617781

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试运行正常，但在请求处理过程中出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94974393459

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617569) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617605) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617612) |
| base-b-test-4-npu-a3 / run (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617669) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868629427/job/94973617773) |


## [Run #31868373873](https://github.com/sgl-project/sglang/actions/runs/31868373873)
- **分支**: `main`
- **总耗时**: 31.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31868373873

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 27.6min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918238) |
| base-b-test-1-npu-a3 / run (0) | 25.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918308) |
| base-b-test-2-npu-a3 / run (0) | 26.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918337) |
| base-b-test-4-npu-a3 / run (1) | 25.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918350) |
| base-b-test-4-npu-a3 / run (0) | 9.2min | 代码错误 | HiCache MLA测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918357) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 25.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918478) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 25.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918479) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94973900899) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94975968402) |

- **base-b-test-16-npu-a3 / run (0)**: 测试在运行第二个测试文件时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918238

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常（HTTP 200），但中途出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918308

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918337

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型加载到55%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918350

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试执行失败（退出码1），耗时270秒，导致整个作业失败。具体失败原因需查看该测试的详细输出，可能是功能实现或测试断言问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918357

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918478

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在gsm8k评估进行到约40%时，出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918479

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示性能测试正常运行中，但突然报错“Executing the custom container implementation failed”，提示联系runner管理员，属于runner或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94973900899

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94975968402

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918201) |
| base-b-test-8-npu-a3 / run (0) | 10.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918243) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918297) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918420) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868373873/job/94972918520) |


## [Run #31868247127](https://github.com/sgl-project/sglang/actions/runs/31868247127)
- **分支**: `agent/whisper-long-audio-chunking`
- **总耗时**: 143.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31868247127

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94973455469) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94975522379) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94977510869) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94985794068) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1097秒。该测试为性能测试，失败可能因性能未达标或环境问题，需查看详细日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94973455469

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业因健康检查检测到同PR中base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94975522379

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94977510869

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94985794068

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611450) |
| base-b-test-2-npu-a3 / run (0) | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611497) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611505) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611507) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611512) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611522) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611561) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611578) |
| base-b-test-16-npu-a3 / run (0) | 48.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611579) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611581) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 123.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611606) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868247127/job/94972611684) |


## [Run #31868231108](https://github.com/sgl-project/sglang/actions/runs/31868231108)
- **分支**: `fix-swa-tombstone-match`
- **总耗时**: 300.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31868231108

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579584) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579619) |
| base-b-test-16-npu-a3 / run (0) | 55.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579637) |
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579639) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579650) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579668) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579677) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579727) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579797) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579806) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579849) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94972579851) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94973036294) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 263.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94975137035) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94976945276) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 77.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31868231108/job/94981531510) |


## [Run #31866988995](https://github.com/sgl-project/sglang/actions/runs/31866988995)
- **分支**: `voidc-minor/jit-moe-topk-softmax`
- **总耗时**: 109.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31866988995

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94970271527) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 54.2min | 性能回归 | NPU性能测试中qwen3_235b_a22b用例失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94972365172) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94973922279) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他性能作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94980745756) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1111秒后退出码1，属于性能测试未通过，可能因模型推理速度未达预期或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94970271527

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行失败（exit code 1），耗时1398秒，其他三个用例均通过，疑似该模型性能未达预期或运行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94972365172

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业未实际运行，因同次运行中另一个作业（base-c-test-perf-8-npu-a3）失败，触发fast-fail跳过，属于依赖作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94973922279

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3和base-c-test-perf-16-npu-a3两个根因失败作业，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94980745756

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 50.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969593876) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969593881) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969593892) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969593903) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969593919) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969593922) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969593930) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969594034) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969594071) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969594083) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866988995/job/94969594109) |


## [Run #31866803987](https://github.com/sgl-project/sglang/actions/runs/31866803987)
- **分支**: `main`
- **总耗时**: 15.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31866803987

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153610) |
| base-b-test-1-npu-a3 / run (0) | 13.2min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153612) |
| base-b-test-16-npu-a3 / run (0) | 10.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153619) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153633) |
| base-b-test-8-npu-a3 / run (0) | 11.2min | 其他 | 作业实际成功，无失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153655) |
| base-b-test-4-npu-a3 / run (1) | 13.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153663) |
| base-b-test-2-npu-a3 / run (0) | 12.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153694) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.3min | 环境问题 | 自定义容器执行失败，NPU测试中途崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153723) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.3min | 环境问题 | 自托管runner执行容器时失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153740) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.4min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153766) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 14.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153785) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153610

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常（进度77%），但runner报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153612

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153619

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试执行失败（exit code 1），耗时282秒，0/5测试通过。可能是代码逻辑错误或环境配置问题导致测试断言失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153633

- **base-b-test-8-npu-a3 / run (0)**: 日志显示所有测试通过（1/1 passed），作业正常结束，仅包含Node.js 20弃用警告，无错误或失败迹象。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153655

- **base-b-test-4-npu-a3 / run (1)**: 日志显示Prefill正常进行，但突然报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153663

- **base-b-test-2-npu-a3 / run (0)**: 日志显示引擎启动成功但随后出现"Executing the custom container implementation failed"错误，可能是容器内NPU设备或环境配置问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153694

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在捕获批次过程中（bs=160）出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，导致测试中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153723

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系runner管理员，属于runner环境或容器执行问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153740

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-4-npu-a3/run(0)，触发fast-fail机制，本作业未实际运行即被跳过，属于CI依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153766

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153785

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866803987/job/94969153616) |


## [Run #31866229885](https://github.com/sgl-project/sglang/actions/runs/31866229885)
- **分支**: `main`
- **总耗时**: 14.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31866229885

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732424) |
| base-b-test-16-npu-a3 / run (0) | 10.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732433) |
| base-b-test-8-npu-a3 / run (0) | 13.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732449) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732462) |
| base-b-test-2-npu-a3 / run (0) | 12.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732488) |
| base-b-test-4-npu-a3 / run (1) | 8.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732501) |
| base-b-test-1-npu-a3 / run (0) | 12.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732528) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.4min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732570) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732580) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732649) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 3.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94968706131) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732424

- **base-b-test-16-npu-a3 / run (0)**: 日志显示服务已成功启动并完成健康检查，但随后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试逻辑失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732433

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732449

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试执行失败，退出码为1，耗时282秒，导致整个作业失败。具体失败原因需查看该测试文件的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732462

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动NPU测试容器时失败，日志显示"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner或容器环境配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732488

- **base-b-test-4-npu-a3 / run (1)**: 日志显示Prefill正常进行，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732501

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732528

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于环境或基础设施问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732570

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试服务正常运行并返回200，但随后出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732580

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然报错“Executing the custom container implementation failed”，随后进入清理流程，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732649

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动测试容器时失败，错误为'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于runner或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94968706131

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732476) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31866229885/job/94967732639) |


## [Run #31865824658](https://github.com/sgl-project/sglang/actions/runs/31865824658)
- **分支**: `main`
- **总耗时**: 9.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31865824658

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.5min | 环境问题 | 作业因环境问题失败，未生成失败产物。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966640986) |
| base-b-test-16-npu-a3 / run (0) | 8.7min | 环境问题 | 自定义容器执行失败，测试在运行中意外终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641132) |
| base-b-test-4-npu-a3 / run (1) | 8.6min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641150) |
| base-b-test-4-npu-a3 / run (0) | 7.7min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641170) |
| base-b-test-2-npu-a3 / run (0) | 7.3min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641175) |
| base-b-test-8-npu-a3 / run (0) | 8.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641179) |
| base-b-test-1-npu-a3 / run (0) | 7.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641329) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641413) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641442) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.6min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641690) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 4.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94967114348) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在运行过程中未产生diffusion-failures目录，上传产物时提示无文件。作业可能因NPU环境配置或资源问题提前终止，未执行到测试完成阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966640986

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试已正常完成（Accuracy 0.955），但随后出现“Executing the custom container implementation failed”错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641132

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行到72%时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU设备或容器环境问题导致作业中断，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641150

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试在捕获批次进行到92%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641170

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在TokenizerManager初始化后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641175

- **base-b-test-8-npu-a3 / run (0)**: 日志显示服务已成功启动，但随后报错“Executing the custom container implementation failed”，提示联系runner管理员，属于NPU CI环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641179

- **base-b-test-1-npu-a3 / run (0)**: 日志显示服务启动成功且生成请求返回200，但随后出现“Executing the custom container implementation failed”错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641329

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动自定义容器时失败，错误信息为“Executing the custom container implementation failed”，可能由于NPU驱动或容器配置问题导致环境无法正常初始化。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641413

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装依赖（evalscope等）过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641442

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641690

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94967114348

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641138) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865824658/job/94966641407) |


## [Run #31865143921](https://github.com/sgl-project/sglang/actions/runs/31865143921)
- **分支**: `pllimax/output-log-dir-structure`
- **总耗时**: 13.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31865143921

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他相关作业已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981046973) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.1min | 环境问题 | 测试脚本启动后立即失败，退出码1，无具体测试日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047012) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047016) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.1min | 环境问题 | 测试脚本启动后立即失败，退出码1，无具体测试日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047018) |

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现base-c-test-acc-8-npu-a3和base-c-test-acc-2-npu-a3两个根因作业失败，触发快速失败机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981046973

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在运行测试命令时立即报错退出，日志中未显示具体测试内容或错误信息，可能是环境初始化或依赖问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047012

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到base-c-test-acc-8-npu-a3和base-c-test-acc-2-npu-a3两个作业失败，作为根因作业，导致本作业被快速失败跳过，并非本作业自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047016

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在运行测试前即报错，错误信息为'command terminated with exit code 1'，但未提供具体失败原因，可能为环境初始化或依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047018

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047187) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047278) |
| base-b-test-16-npu-a3 / run (0) | 52.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047366) |
| base-b-test-4-npu-a3 / run (0) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047375) |
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047384) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047385) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31865143921/job/94981047447) |


## [Run #31864801870](https://github.com/sgl-project/sglang/actions/runs/31864801870)
- **分支**: `main`
- **总耗时**: 22.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31864801870

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 20.9min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100031) |
| base-b-test-4-npu-a3 / run (1) | 18.8min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100037) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 代码错误 | HiCache MLA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100039) |
| multimodal-gen-test-1-npu-a3 | 20.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100040) |
| base-b-test-2-npu-a3 / run (0) | 20.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100073) |
| base-b-test-16-npu-a3 / run (0) | 19.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100111) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100146) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100160) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100178) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964537201) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100031

- **base-b-test-4-npu-a3 / run (1)**: 作业在加载模型分片时（约44%）自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100037

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试文件在NPU A3环境下运行失败，退出码为1，导致整个作业失败。具体失败原因需查看该测试文件的详细输出日志，可能是测试断言失败或运行时错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100039

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未展示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100040

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试运行中（进度81%）时，runner报错“Executing the custom container implementation failed”，随后进入清理流程，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100073

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100111

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，提示联系runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100146

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在执行过程中自定义容器实现失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100160

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100178

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在性能测试运行过程中，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964537201

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964099999) |
| base-b-test-8-npu-a3 / run (0) | 11.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100026) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31864801870/job/94964100099) |


## [Run #31862741908](https://github.com/sgl-project/sglang/actions/runs/31862741908)
- **分支**: `fix/bcg-deepstack-replay-slot`
- **总耗时**: 42.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31862741908

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 20.6min | 代码错误 | NPU DP注意力测试失败，5个测试中仅1个通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858394) |
| base-b-test-16-npu-a3 / run (0) | 28.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858424) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 27.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858621) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 37.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858792) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94961084046) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94962914624) |

- **base-b-test-4-npu-a3 / run (0)**: test_npu_dp_attention.py测试返回退出码1，耗时823秒，导致整个作业失败。该测试涉及DP注意力功能，可能是代码逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858394

- **base-b-test-16-npu-a3 / run (0)**: 日志显示NPU图捕获时NCCL等待警告，随后自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858424

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858621

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858792

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-4-npu-a3/run(0)，触发fast-fail机制，本作业未实际运行即被取消，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94961084046

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-4-npu-a3，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94962914624

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858308) |
| base-b-test-2-npu-a3 / run (0) | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858369) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858378) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858399) |
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858425) |
| base-b-test-4-npu-a3 / run (1) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858478) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858660) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862741908/job/94958858673) |


## [Run #31862528793](https://github.com/sgl-project/sglang/actions/runs/31862528793)
- **分支**: `fix-mxfp4-sharded-state`
- **总耗时**: 93.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31862528793

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958813241) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 6.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94961012407) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败，因另一作业（8-npu）已失败，本作业未实际运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94963349238) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94968225242) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1123秒后退出码1，属于性能测试未通过，可能因模型推理速度未达预期或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958813241

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94961012407

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（4-npu）被级联跳过，未执行实际测试，属于CI级联取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94963349238

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现根因作业 base-c-test-perf-8-npu-a3 失败，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94968225242

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958296931) |
| base-b-test-2-npu-a3 / run (0) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958296943) |
| multimodal-gen-test-1-npu-a3 | 27.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958296951) |
| base-b-test-4-npu-a3 / run (0) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958296954) |
| base-b-test-16-npu-a3 / run (0) | 56.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297001) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297012) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297022) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297036) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297050) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297077) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297086) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862528793/job/94958297112) |


## [Run #31862157648](https://github.com/sgl-project/sglang/actions/runs/31862157648)
- **分支**: `codex/qwen25vl-native-generation`
- **总耗时**: 32.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31862157648

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862157648/job/94957355153) |


## [Run #31862099662](https://github.com/sgl-project/sglang/actions/runs/31862099662)
- **分支**: `xinyuan/nightly-precision-stale-baseline`
- **总耗时**: 132.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31862099662

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 130.8min | 精度回归 | qwen3_5_9b 精度测试失败，其他两个测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204996) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 36.6min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，可能因性能未达预期或运行错误。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94960010041) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b 的 GSM8K 精度测试退出码为 1，而 glm4_7_flash 和 moonlight_16b 均通过，表明该模型存在精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204996

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中1/4通过，qwen3_235b_w8a8_8p_in3k5_out1k5_50ms用例退出码1，耗时1404秒，可能因性能不达标或环境问题导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94960010041

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204793) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204801) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204807) |
| base-b-test-4-npu-a3 / run (0) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204809) |
| base-b-test-4-npu-a3 / run (1) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204834) |
| base-b-test-16-npu-a3 / run (0) | 46.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204851) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204854) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204977) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204989) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957204991) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94957828725) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31862099662/job/94961413695) |


## [Run #31861466139](https://github.com/sgl-project/sglang/actions/runs/31861466139)
- **分支**: `streaming_session`
- **总耗时**: 48.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31861466139

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 2.9min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567658) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567674) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567778) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567784) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.0min | 其他 | 健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567842) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 环境问题 | 健康检查发现lint检查失败，导致作业快速失败，未进入实际测试阶段。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94956735179) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94959931447) |

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567658

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567674

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业未进入实际测试即终止，属于PR代码风格或静态检查问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567778

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567784

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行健康检查时，检测到PR的lint检查结论为failure，触发fast-fail机制，作业未进入实际测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567842

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查状态为failure，触发了fast-fail机制，作业提前终止，未运行NPU性能测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94956735179

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行PR健康检查时，检测到lint检查状态为failure，触发fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94959931447

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567615) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567650) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567685) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567699) |
| base-b-test-4-npu-a3 / run (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567708) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567767) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31861466139/job/94955567890) |


## [Run #31860423783](https://github.com/sgl-project/sglang/actions/runs/31860423783)
- **分支**: `codex/qwen25vl-native-generation`
- **总耗时**: 23.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31860423783

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860423783/job/94953107127) |


## [Run #31860134573](https://github.com/sgl-project/sglang/actions/runs/31860134573)
- **分支**: `main`
- **总耗时**: 107.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31860134573

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 63.0min | 超时 | 测试用例执行超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975630) |
| base-b-test-4-npu-a3 / run (0) | 8.6min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975671) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 103.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975822) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.6min | 性能回归 | NPU性能测试未达预期，minimax_m2_5测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94952825865) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94955110343) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94956872529) |

- **base-b-test-1-npu-a3 / run (0)**: TestAscendSamplingBackend.test_mmlu 测试从02:58开始执行，直到03:54才结束，耗时约56分钟，接近60分钟的超时限制，最终容器执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975630

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试执行失败（退出码1），耗时293秒，导致整个作业失败。具体失败原因需查看该测试的详细输出，可能是功能实现或测试断言问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975671

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行约1小时43分钟后，在Decode阶段正常输出时突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975822

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1144秒后失败，该测试为性能测试，可能因性能未达标或超时导致退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94952825865

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，导致本作业在启动前被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94955110343

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现根因失败作业为base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94956872529

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975533) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975604) |
| base-b-test-8-npu-a3 / run (0) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975653) |
| base-b-test-4-npu-a3 / run (1) | 27.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975663) |
| base-b-test-16-npu-a3 / run (0) | 80.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975670) |
| base-b-test-2-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975672) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975789) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975818) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860134573/job/94951975891) |


## [Run #31860091936](https://github.com/sgl-project/sglang/actions/runs/31860091936)
- **分支**: `codex/honor-explicit-model-loader`
- **总耗时**: 52.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31860091936

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916038) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因同PR中另一个作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916229) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 环境问题 | K8s Pod启动失败，作业未开始即终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916257) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因同批次其他作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916286) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业根因失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94956550629) |

- **base-b-test-2-npu-a3 / run (0)**: 该作业在启动时执行健康检查，检测到同一PR中base-c-test-acc-16-npu-a3作业已失败，被判定为根因失败，因此本作业被跳过（fast-fail），并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916038

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-acc-16-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916229

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: Runner尝试启动K8s Pod时，Pod状态变为Failed且不健康，导致作业在初始化阶段失败。日志显示Pod创建后未能正常上线，属于基础设施/环境问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916257

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示本作业未实际运行测试，而是因PR健康检查检测到同run中base-c-test-acc-16-npu-a3作业失败，触发fast-fail机制，导致本作业被跳过并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916286

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现base-c-test-acc-16-npu-a3作业为根因失败，本作业作为级联失败被快速跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94956550629

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951915994) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916014) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916035) |
| base-b-test-16-npu-a3 / run (0) | 51.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916053) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916055) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916064) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916083) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31860091936/job/94951916191) |


## [Run #31859834635](https://github.com/sgl-project/sglang/actions/runs/31859834635)
- **分支**: `dsv4_fp8_trtllm_gen`
- **总耗时**: 79.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31859834635

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 75.0min | 精度回归 | Qwen3.5-9B GSM8K 精度测试失败，0/3 用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152824) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94952198132) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | PR健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94954576976) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查失败：lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94956073935) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 test_npu_qwen3_5_9b_bf16_1p_gsm8k.py 运行 4240 秒后退出码为 1，所有 3 个精度用例均未通过，表明模型输出精度不达标，可能由代码改动或环境差异引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152824

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段执行健康检查时，发现lint检查失败（conclusion=failure），触发fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94952198132

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查状态为failure，触发fast-fail机制，作业未进入实际测试即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94954576976

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查状态为failure，触发了fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94956073935

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152507) |
| base-b-test-4-npu-a3 / run (0) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152553) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152557) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152560) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152596) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152609) |
| base-b-test-2-npu-a3 / run (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152623) |
| base-b-test-16-npu-a3 / run (0) | 52.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152676) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152716) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152841) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859834635/job/94951152967) |


## [Run #31859829531](https://github.com/sgl-project/sglang/actions/runs/31859829531)
- **分支**: `feat/ltx-2.5-support`
- **总耗时**: 22.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31859829531

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31859829531/job/94951128755) |


---
*Auto-generated by npu_pr_monitor.py*