# NPU CI 执行监控
**生成时间**: 2026-08-17 12:36 UTC
**分析 Run 数**: 44

---

## 📊 本次执行总结

- **成功 Job 数**: 38
- **失败 Run 数**: 44
- **成功 Job 平均耗时**: 11.0min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-b-test-16-npu-a3 / run (0) | 48.4min | #32011314757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384478) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.1min | #32011314757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384794) |
| base-b-test-4-npu-a3 / run (0) | 31.6min | #32011314757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384330) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.7min | #32011314757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384915) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | #32011314757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384310) |
| base-b-test-2-npu-a3 / run (0) | 22.5min | #32011314757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384452) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.6min | #32011314757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95360209227) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | #32011314757 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384332) |
| base-a-test-1-npu-a2 / run (0) | 8.5min | #32011248093 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194003) |
| base-a-test-1-npu-a2 / run (0) | 8.4min | #32009519713 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995860) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 41 | 0 | 33 | 0 | 8 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 41 | 0 | 0 | 29 | 12 |
| base-b-test-16-npu-a3 / run (0) | 41 | 1 | 0 | 28 | 12 |
| base-b-test-1-npu-a3 / run (0) | 41 | 1 | 0 | 28 | 12 |
| base-b-test-8-npu-a3 / run (0) | 41 | 1 | 0 | 28 | 12 |
| base-b-test-4-npu-a3 / run (1) | 41 | 1 | 0 | 28 | 12 |
| base-b-test-4-npu-a3 / run (0) | 41 | 1 | 0 | 28 | 12 |
| base-b-test-2-npu-a3 / run (0) | 41 | 1 | 0 | 28 | 12 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 41 | 1 | 0 | 28 | 12 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 41 | 1 | 0 | 28 | 12 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41 | 1 | 0 | 28 | 12 |
| base-a-test-1-npu-a2 / run (0) | 41 | 28 | 0 | 4 | 9 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1 | 0 | 0 | 1 | 0 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1 | 0 | 0 | 1 | 0 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32011314757<br>[#32162 [HiSparse] Support hisparse multi-step swap io kernel](https://github.com/sgl-project/sglang/pull/32162) | `hisparse_mtp_kernel` | 171.9min | 0 |  | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32011314757) |
| #32007903752<br>[#30398 [Refactor] New EPD](https://github.com/sgl-project/sglang/pull/30398) | `new_epd` | 145.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007903752) |
| #32007034012 | `fix/xpu-decode-graph-runner-is-current-stream-capturing` | 143.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007034012) |
| #32007863939<br>[#34406 TP/PP Consensus checker](https://github.com/sgl-project/sglang/pull/34406) | `consensus_checker_0806` | 143.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007863939) |
| #32007288787<br>[#30575 [AMD] Enable Fast Triton Sparse MLA backend](https://github.com/sgl-project/sglang/pull/30575) | `feat/triton-sparse-mla` | 142.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007288787) |
| #32007504074<br>[#34580 [AMD] Optimize KIMI-K3 with Triton MLA decode kernel by tuning the stage-1 geometry for gfx950](https://github.com/sgl-project/sglang/pull/34580) | `amd-mla-decode-gfx950-tune` | 142.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007504074) |
| #32007532052<br>[#33576  [AMD] Add Work-Centric (Lean) Attention: a persistent-CTA decode kernel for long-context serving](https://github.com/sgl-project/sglang/pull/33576) | `wca-rebased` | 142.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007532052) |
| #32007820836<br>[#30805 [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103](https://github.com/sgl-project/sglang/pull/30805) | `dsv4_fp8_trtllm_gen` | 141.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007820836) |
| #32007270370<br>[#24911 Profiling Enhancements [2/3]: detailed execution step annotations](https://github.com/sgl-project/sglang/pull/24911) | `feat/roofline_annotations` | 140.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007270370) |
| #32007823930<br>[#35081 Using unified radix tree by default for all case](https://github.com/sgl-project/sglang/pull/35081) | `hybrid_tree/change-default` | 140.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32007823930) |
| #32006924715<br>[#33561 [Model] Support Ling-3.0-flash (BailingMoeV3) ](https://github.com/sgl-project/sglang/pull/33561) | `ling3-flash-dspark` | 140.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32006924715) |
| #32006713528<br>[#35004 [Diffusion] Reuse SRT CLIP encoder blocks](https://github.com/sgl-project/sglang/pull/35004) | `codex/diffusion-reuse-srt-clip` | 138.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32006713528) |
| #32008637548<br>[#33473 [HiCache] Batch PP write and load completion sync](https://github.com/sgl-project/sglang/pull/33473) | `bytedance/hicache-batch-pp-completions` | 137.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32008637548) |
| #32008686153<br>[#35060 Clean up environ.py: remove dead env vars, unify deprecation handling, move examples to a unit test](https://github.com/sgl-project/sglang/pull/35060) | `cleanup-environ` | 137.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32008686153) |
| #32008616124<br>[#34888 Split TRTLLM MHA decode batches by KV sequence length](https://github.com/sgl-project/sglang/pull/34888) | `yangminl/trtllm-mha-ragged-split2` | 135.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32008616124) |
| #32009519713<br>[#12961 Fix DP attention on CPU](https://github.com/sgl-project/sglang/pull/12961) | `chunyuan/pr_dp_attention` | 131.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32009519713) |
| #32009425439<br>[#32856 [CPU] Fix NUMA/core binding for DP ranks](https://github.com/sgl-project/sglang/pull/32856) | `chunyuan/pr_dp_fix` | 131.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32009425439) |
| #32009316141<br>[#30144 [XPU] Enable fused GDN QKV split Triton kernel on XPU](https://github.com/sgl-project/sglang/pull/30144) | `jiayi/GDN` | 131.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32009316141) |
| #32011699740<br>[#32313 [Feature] Optimize TP LMHead with All-to-All](https://github.com/sgl-project/sglang/pull/32313) | `lm-head-opt` | 125.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32011699740) |
| #32010274092<br>[#31370 feat(moe): fold padded-topk_ids fill into fused shared-experts append+remap](https://github.com/sgl-project/sglang/pull/31370) | `feat/fold-pad-fill-into-moe-append-remap` | 124.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32010274092) |
| #32010474266<br>[#34142 Fix inflated row pitch when a CP round-robin shard has a single row](https://github.com/sgl-project/sglang/pull/34142) | `fix/dsa-cp-round-robin-single-row-shard-stride` | 123.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32010474266) |
| #32010423138<br>[#30565 [AMD] [GLM5] Fix MTP layer_quant_config in-place mutation + add nextn Quark-exclude unit test](https://github.com/sgl-project/sglang/pull/30565) | `tmp/eagle-mtp` | 122.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32010423138) |
| #32011453346<br>[#34798 [HiCache] Buffer-only mode for HiCache host memory layer](https://github.com/sgl-project/sglang/pull/34798) | `hicache-buffer-only-mode` | 117.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32011453346) |
| #32012336980<br>[#34926 Clean deprecated DeepSeek V4 Environs](https://github.com/sgl-project/sglang/pull/34926) | `clean-dsv4` | 116.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32012336980) |
| #32012149499<br>[#34821 [GDN] fix: use triton causal_conv1d_update for target_verify path](https://github.com/sgl-project/sglang/pull/34821) | `jiayi/fix_bug` | 116.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32012149499) |
| #32012048641<br>[#34715 [bugfix] [NPU] fix transpose batch matmul K*B exceed 65536.](https://github.com/sgl-project/sglang/pull/34715) | `bmm65536-fallback` | 116.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32012048641) |
| #32011248093<br>[#33829 [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding](https://github.com/sgl-project/sglang/pull/33829) | `dots-note-for-sglang` | 116.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32011248093) |
| #32012471189<br>[#35126 [Spec] Stage EAGLE draft-extend graph inputs before the verify launch](https://github.com/sgl-project/sglang/pull/35126) | `lsyin/draft-extend-input-staging` | 113.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32012471189) |
| #32011984891<br>[#34923 Apply latest DeepEP branch](https://github.com/sgl-project/sglang/pull/34923) | `codex/deepep-nvshmem-qp-depth` | 113.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32011984891) |
| #32011284470<br>[#35114 [kernels] Reorganize ops/diffusion by operator domain behind a lazy facade](https://github.com/sgl-project/sglang/pull/35114) | `diffusion-kernels-reorg` | 96.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32011284470) |
| #32011629471 | `cheng/gc-s12-carrier` | 78.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32011629471) |
| #32009322315<br>[#34695 [AMD] Speed up Wan2.2 DiT FP8 attention per-tensor quantization](https://github.com/sgl-project/sglang/pull/34695) | `amd/wan22-fp8-pertensor-quant` | 63.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32009322315) |
| #32009315233<br>[#34424 [AMD] Fix ROCm VAE Conv2D fast path breaking spatial-parallel decode](https://github.com/sgl-project/sglang/pull/34424) | `amd/fix-vae-spatial-parallel-decode-rocm-conv2d` | 57.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32009315233) |
| #32010840469<br>[#30519 [AMD] [GLM5] fp8 MLA absorbed bmm for GLM-5.2 on gfx950](https://github.com/sgl-project/sglang/pull/30519) | `jacob/glm-mla-fp8-absorbed-bmm` | 43.4min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32010840469) |
| #32011898635<br>[#34197 [diffusion] RL rollout support for the Cosmos3 pipeline](https://github.com/sgl-project/sglang/pull/34197) | `feat/fused-lora-shards` | 38.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32011898635) |
| #32008339065<br>[#34565 [Unified Tree] Support Branching-Point Caching for the SWA Component](https://github.com/sgl-project/sglang/pull/34565) | `cjc/unified-swa-branching` | 22.6min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32008339065) |
| #32007855113<br>[#33685 [NPU CI] Reorganize test output/log directory structure with workflow context](https://github.com/sgl-project/sglang/pull/33685) | `pllimax/output-log-dir-structure` | 17.8min | 11 | base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32007855113) |
| #32012097270<br>[#35070 [PD] Avoid unused PREBUILT prompt tensor transfer](https://github.com/sgl-project/sglang/pull/35070) | `main` | 14.3min | 12 | base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32012097270) |
| #32006905491<br>[#33685 [NPU CI] Reorganize test output/log directory structure with workflow context](https://github.com/sgl-project/sglang/pull/33685) | `pllimax/output-log-dir-structure` | 13.2min | 10 | base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32006905491) |
| #32011099839<br>[#35059 [Spec] Resolve shared-read ends from the backend declaration alone](https://github.com/sgl-project/sglang/pull/35059) | `main` | 12.8min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32011099839) |
| #32011148579<br>[#34923 Apply latest DeepEP branch](https://github.com/sgl-project/sglang/pull/34923) | `codex/deepep-nvshmem-qp-depth` | 10.8min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32011148579) |
| #32006710139 | `new_epd` | 10.3min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32006710139) |
| #32010485744<br>[#33676 [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management](https://github.com/sgl-project/sglang/pull/33676) | `main` | 7.9min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32010485744) |
| #32011465857<br>[#34715 [bugfix] [NPU] fix transpose batch matmul K*B exceed 65536.](https://github.com/sgl-project/sglang/pull/34715) | `bmm65536-fallback` | 7.5min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-b-test-4-npu-a3 / run (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32011465857) |

---


## [Run #32012471189](https://github.com/sgl-project/sglang/actions/runs/32012471189)
- **分支**: `lsyin/draft-extend-input-staging`
- **总耗时**: 113.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32012471189

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.5min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610762) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610840) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610895) |
| base-b-test-8-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610905) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610934) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611009) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611028) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611314) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | 级联失败，根因是其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611328) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611331) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611336) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出，仅显示Node 20弃用警告和上传diffusion-failures目录时未找到文件。无法判断测试是否失败，可能因日志截断或测试未运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610762

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610840

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610895

- **base-b-test-8-npu-a3 / run (0)**: 日志显示本作业未实际运行测试，而是在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，随后触发fast-fail跳过本作业，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610905

- **base-b-test-4-npu-a3 / run (1)**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被过滤后，因根因作业失败而触发fast-fail机制，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610934

- **base-b-test-4-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，被识别为根因失败，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611009

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，将其视为根因，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611028

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611314

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示本作业因健康检查过滤了级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业并非直接失败，而是被级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611328

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611331

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335611336

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32012471189/job/95335610856) |


## [Run #32012336980](https://github.com/sgl-project/sglang/actions/runs/32012336980)
- **分支**: `clean-dsv4`
- **总耗时**: 116.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32012336980

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 41.4min | 环境问题 | 作业因环境问题失败，未生成失败产物。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505442) |
| base-b-test-16-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505658) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505787) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505813) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505860) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505865) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505918) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334506277) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 作业因健康检查检测到其他根因作业失败而被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334506435) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334506457) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334506484) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在运行过程中未产生diffusion-failures目录，上传产物时提示无文件。作业可能因NPU环境配置或资源问题提前终止，未完成测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505442

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505658

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505787

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505813

- **base-b-test-4-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，导致本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505860

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505865

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505918

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334506277

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示该作业本身未执行测试，而是被健康检查机制识别为级联失败，根因作业为multimodal-gen-test-1-npu-a3，导致本作业被跳过并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334506435

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段因健康检查检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334506457

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334506484

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32012336980/job/95334505767) |


## [Run #32012149499](https://github.com/sgl-project/sglang/actions/runs/32012149499)
- **分支**: `jiayi/fix_bug`
- **总耗时**: 116.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32012149499

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.8min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942335) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942436) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他作业根因失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942437) |
| base-b-test-8-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942446) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因失败作业，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942483) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942558) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943024) |
| base-b-test-4-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业根因失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943082) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943092) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.0min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943154) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943338) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在运行测试后上传diffusion-failures工件时未找到文件，但未展示测试执行过程或失败断言，无法判断具体失败原因，可能为测试通过但作业被外部中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942335

- **base-b-test-2-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败作业，因此本作业（base-b-test-2-npu-a3）被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942436

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942437

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942446

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因失败为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942483

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业被跳过未实际运行，属于依赖作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942558

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因Fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943024

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤后根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943082

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943092

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943154

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333943338

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32012149499/job/95333942470) |


## [Run #32012097270](https://github.com/sgl-project/sglang/actions/runs/32012097270)
- **分支**: `main`
- **总耗时**: 14.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32012097270

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789516) |
| base-b-test-1-npu-a3 / run (0) | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789584) |
| multimodal-gen-test-1-npu-a3 | 13.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789586) |
| base-b-test-8-npu-a3 / run (0) | 13.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789610) |
| base-b-test-4-npu-a3 / run (1) | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789643) |
| base-b-test-16-npu-a3 / run (0) | 13.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789681) |
| base-b-test-2-npu-a3 / run (0) | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789691) |
| base-b-test-4-npu-a3 / run (0) | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789722) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789863) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789864) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789889) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 13.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333790015) |

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789516

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件已被删除或路径错误，属于基础设施或配置问题，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789584

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的依赖文件或数据在存储账户中缺失，可能是文件被误删、路径配置错误或上传未完成，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789586

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明作业尝试下载的blob（可能为模型权重或测试数据）在存储中不存在，可能是路径错误、文件被删除或上传未完成，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789610

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789643

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789681

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789691

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789722

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789863

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，或路径配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789864

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333789889

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012097270/job/95333790015


## [Run #32012048641](https://github.com/sgl-project/sglang/actions/runs/32012048641)
- **分支**: `bmm65536-fallback`
- **总耗时**: 116.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32012048641

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 49.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678790) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678832) |
| base-b-test-4-npu-a3 / run (0) | 0.6min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678907) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678937) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678972) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679105) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679127) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679225) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | 健康检查失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679235) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679331) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.0min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679397) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业整体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678790

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678832

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678907

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-2-npu-a3）被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678937

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业（base-b-test-4-npu-a3）因级联失败被过滤并快速失败，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678972

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679105

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，触发fast-fail跳过本作业，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679127

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679225

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业因PR健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，被快速失败机制跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679235

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679331

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤并快速失败，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333679397

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32012048641/job/95333678900) |


## [Run #32011984891](https://github.com/sgl-project/sglang/actions/runs/32011984891)
- **分支**: `codex/deepep-nvshmem-qp-depth`
- **总耗时**: 113.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011984891

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434264) |
| base-b-test-1-npu-a3 / run (0) | 1.6min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434379) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434394) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434411) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434479) |
| base-b-test-4-npu-a3 / run (1) | 1.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434486) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434534) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434618) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 该作业因健康检查检测到其他根因作业失败而被快速失败跳过，并非自身测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434752) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434834) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434836) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434911) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未包含任何测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434264

- **base-b-test-1-npu-a3 / run (0)**: 该作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制跳过本作业，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434379

- **base-a-test-1-npu-a2 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，将其判定为根因失败，因此本作业（base-a-test-1-npu-a2）被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434394

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查识别出根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434411

- **base-b-test-16-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而被快速失败跳过，并非自身测试失败，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434479

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434486

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434534

- **base-b-test-2-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，被识别为根因，导致本作业快速失败，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434618

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，导致本作业被Fast-fail跳过，属于级联失败而非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434752

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434834

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434836

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业因PR测试健康检查机制被跳过，根因是多模态生成测试（multimodal-gen-test-1-npu-a3）失败，本作业作为级联失败被过滤，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011984891/job/95333434911


## [Run #32011898635](https://github.com/sgl-project/sglang/actions/runs/32011898635)
- **分支**: `feat/fused-lora-shards`
- **总耗时**: 38.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011898635

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 23.4min | 环境问题 | Git 拉取代码失败，远端仓库缺少指定 commit 引用。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011898635/job/95361934150) |

- **multimodal-gen-test-1-npu-a3**: 作业在 checkout 阶段执行 git fetch 时，远端返回 "not our ref"，多次重试均失败，导致无法获取 PR 合并提交，最终作业退出。可能是 PR 已更新或缓存不一致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011898635/job/95361934150


## [Run #32011699740](https://github.com/sgl-project/sglang/actions/runs/32011699740)
- **分支**: `lm-head-opt`
- **总耗时**: 125.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011699740

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 47.7min | 其他 | 作业日志被截断，未显示实际测试结果，仅见上传工件时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570677) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570798) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因作业失败，导致级联跳过当前作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570801) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570820) |
| base-b-test-1-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570847) |
| base-b-test-4-npu-a3 / run (1) | 2.0min | 其他 | 健康检查发现根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570959) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570972) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | PR健康检查失败，因其他根因作业失败导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332571038) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332571058) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332571151) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.1min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332571243) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法判断具体失败原因。仅能看到上传diffusion-failures目录时提示无文件，可能测试未产生失败样本或测试未执行。需查看完整日志以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570677

- **base-b-test-8-npu-a3 / run (0)**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因健康检查过滤后判定为级联失败，未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570798

- **base-b-test-16-npu-a3 / run (0)**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，当前作业因级联失败被过滤并快速失败，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570801

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570820

- **base-b-test-1-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，判定为根因作业，因此本作业（base-b-test-1-npu-a3）被快速失败跳过，并非自身执行出错。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570847

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570959

- **base-b-test-2-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，被判定为根因，本作业因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570972

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332571038

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，因此本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332571058

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业作为依赖被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332571151

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因Fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332571243

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011699740/job/95332570774) |


## [Run #32011629471](https://github.com/sgl-project/sglang/actions/runs/32011629471)
- **分支**: `cheng/gc-s12-carrier`
- **总耗时**: 78.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011629471

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447907) |
| base-b-test-1-npu-a3 / run (0) | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447963) |
| base-b-test-2-npu-a3 / run (0) | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447989) |
| base-b-test-16-npu-a3 / run (0) | 77.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447999) |
| base-b-test-8-npu-a3 / run (0) | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448020) |
| base-b-test-4-npu-a3 / run (0) | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448023) |
| base-b-test-4-npu-a3 / run (1) | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448055) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448414) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448499) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448513) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448586) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含multimodal-gen测试的实际执行输出或错误信息，仅有GitHub Actions的常规准备、上传artifact（无文件）和清理步骤，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447907

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447963

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447989

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是因为日志文件被清理、路径错误或上传失败，属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447999

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448020

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448023

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448055

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448414

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448499

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448513

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332448586

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011629471/job/95332447885) |


## [Run #32011465857](https://github.com/sgl-project/sglang/actions/runs/32011465857)
- **分支**: `bmm65536-fallback`
- **总耗时**: 7.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011465857

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842103) |
| base-b-test-1-npu-a3 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842126) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842152) |
| base-b-test-16-npu-a3 / run (0) | 7.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842205) |
| base-b-test-4-npu-a3 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842227) |
| base-b-test-2-npu-a3 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842248) |
| base-a-test-1-npu-a2 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842354) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842534) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842545) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842583) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842627) |
| base-b-test-4-npu-a3 / run (1) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842637) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842103

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842126

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842152

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径配置错误，或资源被清理。属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842205

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842227

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842248

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842354

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842534

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842545

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842583

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842627

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011465857/job/95331842637


## [Run #32011453346](https://github.com/sgl-project/sglang/actions/runs/32011453346)
- **分支**: `hicache-buffer-only-mode`
- **总耗时**: 117.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011453346

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 0.7min | 其他 | 健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832621) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832650) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 环境问题 | PR健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832661) |
| base-b-test-8-npu-a3 / run (0) | 1.3min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832685) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832707) |
| base-b-test-16-npu-a3 / run (0) | 2.3min | 其他 | 健康检查中的lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832737) |
| multimodal-gen-test-1-npu-a3 | 40.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832741) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832769) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | PR健康检查中lint检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331833048) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.0min | 其他 | 健康检查失败：lint检查未通过导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331833117) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查失败：lint检查未通过导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331833118) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | 健康检查失败：lint检查未通过导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331833119) |

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发fast-fail机制，作业被终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832621

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行lint检查时失败（conclusion=failure），触发fast-fail机制，作业未进入实际测试即终止。这是代码风格或静态检查问题，非NPU测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832650

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业未进入实际测试即终止。这是PR代码风格或格式问题，而非NPU测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832661

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint 检查结论为 failure，触发 fast-fail 机制，作业立即失败退出，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832685

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行lint检查时，检查状态为failure，触发了fast-fail机制，导致作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832707

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行lint检查时失败（conclusion=failure），触发了fast-fail机制，导致整个作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832737

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832741

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行lint健康检查时失败（conclusion=failure），触发了fast-fail机制，导致整个作业在运行测试前就被终止，属于前置检查未通过而非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331832769

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行health-check时，检测到该PR的lint检查结论为failure，触发了fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331833048

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331833117

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331833118

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行健康检查时，检测到PR的lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011453346/job/95331833119


## [Run #32011314757](https://github.com/sgl-project/sglang/actions/runs/32011314757)
- **分支**: `hisparse_mtp_kernel`
- **总耗时**: 171.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011314757

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.4min | 精度回归 | moonshotai_moonlight_16b_a3b 测试失败，导致整体作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384752) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95364926911) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 3.4min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95369071386) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 1/3 通过，qwen3_5_9b 通过，但 moonshotai_moonlight_16b_a3b 在 32 秒内退出（exit code 1），可能因模型加载或精度问题导致，需进一步查看该测试日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384752

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-acc-2-npu-a3 作业失败，被判定为根因失败，导致本作业（base-c-test-perf-16-npu-a3）在启动前被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95364926911

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-acc-2-npu-a3 作业失败，本作业作为级联失败被过滤，最终因根因作业失败而快速失败，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95369071386

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384251) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384310) |
| base-b-test-4-npu-a3 / run (0) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384330) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384332) |
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384439) |
| base-b-test-2-npu-a3 / run (0) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384452) |
| base-b-test-16-npu-a3 / run (0) | 48.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384478) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384792) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384794) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95331384915) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011314757/job/95360209227) |


## [Run #32011284470](https://github.com/sgl-project/sglang/actions/runs/32011284470)
- **分支**: `diffusion-kernels-reorg`
- **总耗时**: 96.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011284470

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 52.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301175) |
| base-b-test-16-npu-a3 / run (0) | 96.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301265) |
| base-b-test-1-npu-a3 / run (0) | 96.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301274) |
| base-b-test-2-npu-a3 / run (0) | 96.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301311) |
| base-b-test-8-npu-a3 / run (0) | 96.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301316) |
| base-b-test-4-npu-a3 / run (0) | 96.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301342) |
| base-b-test-4-npu-a3 / run (1) | 96.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301500) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 96.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301852) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 96.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301906) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 96.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301943) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 96.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331302044) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，仅显示上传diffusion-failures目录时无文件，无法判断具体失败原因，可能为测试未产生失败文件或日志不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301175

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301265

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301274

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是上传失败、路径错误或资源被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301311

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301316

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或依赖文件在 Azure Blob 存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301342

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301500

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301852

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301906

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301943

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331302044

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011284470/job/95331301207) |


## [Run #32011248093](https://github.com/sgl-project/sglang/actions/runs/32011248093)
- **分支**: `dots-note-for-sglang`
- **总耗时**: 116.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011248093

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 52.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331193902) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331193949) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331193955) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194039) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194080) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194099) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194112) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194826) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.5min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194895) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 作业因健康检查被跳过，实际根因是其他作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194937) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.3min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194945) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331193902

- **base-b-test-4-npu-a3 / run (1)**: 作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331193949

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，主动跳过执行，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331193955

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被终止，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194039

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194080

- **base-b-test-2-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，触发Fast-fail机制，本作业（base-b-test-2-npu-a3）被跳过未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194099

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194112

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查识别出根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194826

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败后，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194895

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业在启动前被健康检查过滤，因根因作业multimodal-gen-test-1-npu-a3失败而快速失败，自身未执行测试，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194937

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194945

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 8.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32011248093/job/95331194003) |


## [Run #32011148579](https://github.com/sgl-project/sglang/actions/runs/32011148579)
- **分支**: `codex/deepep-nvshmem-qp-depth`
- **总耗时**: 10.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011148579

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864129) |
| base-b-test-2-npu-a3 / run (0) | 10.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864237) |
| base-b-test-8-npu-a3 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864309) |
| base-b-test-16-npu-a3 / run (0) | 10.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864326) |
| base-b-test-1-npu-a3 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864344) |
| base-b-test-4-npu-a3 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864369) |
| base-b-test-4-npu-a3 / run (1) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864405) |
| base-a-test-1-npu-a2 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864610) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864986) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330865187) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330865213) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330865282) |

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是CI脚本引用了不存在的文件或路径配置错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864129

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864237

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864309

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储账户中缺失，可能是上传失败、路径错误或资源被清理，需检查上游作业或存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864326

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查 blob 路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864344

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864369

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864405

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置变更导致，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864610

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330864986

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330865187

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330865213

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011148579/job/95330865282


## [Run #32011099839](https://github.com/sgl-project/sglang/actions/runs/32011099839)
- **分支**: `main`
- **总耗时**: 12.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32011099839

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716574) |
| base-b-test-4-npu-a3 / run (0) | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716675) |
| base-b-test-1-npu-a3 / run (0) | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716692) |
| base-a-test-1-npu-a2 / run (0) | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716700) |
| base-b-test-4-npu-a3 / run (1) | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716751) |
| base-b-test-2-npu-a3 / run (0) | 12.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716773) |
| base-b-test-16-npu-a3 / run (0) | 12.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716799) |
| base-b-test-8-npu-a3 / run (0) | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716847) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330717088) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330717130) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330717171) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330717334) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716574

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716675

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716692

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716700

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716751

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716773

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716799

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330716847

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、缓存或构建产物）在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330717088

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330717130

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330717171

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是文件被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32011099839/job/95330717334


## [Run #32010840469](https://github.com/sgl-project/sglang/actions/runs/32010840469)
- **分支**: `jacob/glm-mla-fp8-absorbed-bmm`
- **总耗时**: 43.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32010840469

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 27.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传artifact时无失败文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942201) |
| base-b-test-16-npu-a3 / run (0) | 42.7min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942212) |
| base-b-test-1-npu-a3 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942254) |
| base-b-test-2-npu-a3 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942259) |
| base-a-test-1-npu-a2 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942270) |
| base-b-test-4-npu-a3 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942306) |
| base-b-test-4-npu-a3 / run (1) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942343) |
| base-b-test-8-npu-a3 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942358) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942598) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942612) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942634) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942727) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到具体测试输出或错误信息。最终上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记为失败，需查看完整日志定位真实原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942201

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的blob（可能为测试数据或模型权重）已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942212

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942254

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942259

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942270

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942306

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942343

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942358

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942598

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942612

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查存储路径和资源准备步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942634

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010840469/job/95329942727


## [Run #32010485744](https://github.com/sgl-project/sglang/actions/runs/32010485744)
- **分支**: `main`
- **总耗时**: 7.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32010485744

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328865995) |
| base-b-test-2-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866105) |
| base-b-test-1-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866140) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866154) |
| base-b-test-4-npu-a3 / run (1) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866160) |
| base-a-test-1-npu-a2 / run (0) | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866201) |
| base-b-test-4-npu-a3 / run (0) | 7.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866263) |
| base-b-test-16-npu-a3 / run (0) | 7.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866350) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866446) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866474) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866547) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866565) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328865995

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866105

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储中的文件不存在，可能是由于文件被删除、路径错误或上传失败，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866140

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866154

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866160

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866201

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖缺失，需检查CI配置中的存储路径或资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866263

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI依赖的远程存储对象缺失或路径错误，可能是上传失败、清理策略或配置变更所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866350

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866446

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866474

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866547

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010485744/job/95328866565


## [Run #32010474266](https://github.com/sgl-project/sglang/actions/runs/32010474266)
- **分支**: `fix/dsa-cp-round-robin-single-row-shard-stride`
- **总耗时**: 123.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32010474266

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.6min | 其他 | 作业未显示明确失败原因，仅上传artifact时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869700) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869731) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869733) |
| base-b-test-4-npu-a3 / run (1) | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869772) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869777) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，导致级联跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869808) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869831) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869893) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业根因失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328870055) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.3min | 其他 | PR测试健康检查失败，根因是多模态生成测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328870098) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 作业因其他根因作业失败被快速失败机制跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328870112) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328870133) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业正常执行，最后上传diffusion-failures目录时提示无文件，未发现测试失败或错误信息，可能为作业提前结束或测试未产生失败文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869700

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869731

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869733

- **base-b-test-4-npu-a3 / run (1)**: 该作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，因此被跳过（fast-fail），并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869772

- **base-b-test-4-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，未实际运行测试即退出，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869777

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869808

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，未实际运行测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869831

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328869893

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328870055

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328870098

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业被级联过滤后快速失败，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328870112

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业在“Check PR test health”步骤被快速失败机制跳过，根因作业为multimodal-gen-test-1-npu-a3，本作业并非实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010474266/job/95328870133


## [Run #32010423138](https://github.com/sgl-project/sglang/actions/runs/32010423138)
- **分支**: `tmp/eagle-mtp`
- **总耗时**: 122.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32010423138

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 41.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686016) |
| base-b-test-16-npu-a3 / run (0) | 1.4min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686138) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686139) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686223) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686265) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686303) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686354) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686388) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | PR健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686663) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686714) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686780) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.4min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686931) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686016

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686138

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686139

- **base-b-test-8-npu-a3 / run (0)**: 该作业因其他作业（multimodal-gen-test-1-npu-a3）失败而被级联取消，并非自身问题。日志显示健康检查过滤了多个级联失败，最终根因是multimodal-gen-test-1-npu-a3，本作业被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686223

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-a-test-1-npu-a2）被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686265

- **base-b-test-2-npu-a3 / run (0)**: 该作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，导致本作业被跳过并报错，属于级联失败而非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686303

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686354

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686388

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686663

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686714

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686780

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010423138/job/95328686931


## [Run #32010274092](https://github.com/sgl-project/sglang/actions/runs/32010274092)
- **分支**: `feat/fold-pad-fill-into-moe-append-remap`
- **总耗时**: 124.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32010274092

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286462) |
| base-b-test-16-npu-a3 / run (0) | 1.6min | 其他 | 健康检查快速失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286614) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业根因失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286615) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他作业根因失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286622) |
| base-b-test-4-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286665) |
| base-b-test-1-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286743) |
| base-b-test-4-npu-a3 / run (1) | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286760) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286802) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业（multimodal-gen-test-1-npu-a3）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286807) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，因同一次运行中另一个作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286834) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286913) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures工件时未找到文件，说明测试可能未产生失败样本，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286462

- **base-b-test-16-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286614

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因Fast-fail机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286615

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286622

- **base-b-test-4-npu-a3 / run (0)**: 健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286665

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，本作业因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286743

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动前的健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286760

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因失败为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286802

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 本作业在启动前的PR健康检查阶段被快速失败（fast-fail），原因是根因作业multimodal-gen-test-1-npu-a3已失败，本作业被级联跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286807

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286834

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业在启动前执行PR测试健康检查，检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286913

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32010274092/job/95328286641) |


## [Run #32009519713](https://github.com/sgl-project/sglang/actions/runs/32009519713)
- **分支**: `chunyuan/pr_dp_attention`
- **总耗时**: 131.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32009519713

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 41.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995753) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995836) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995897) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995929) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995978) |
| base-b-test-16-npu-a3 / run (0) | 1.4min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996013) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996442) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现根因任务失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996656) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996686) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996745) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996758) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最终上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995753

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995836

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995897

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995929

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3为根因失败，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995978

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996013

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996442

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败任务，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996656

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996686

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996745

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查识别出根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤，最终因快速失败策略退出，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325996758

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 8.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32009519713/job/95325995860) |


## [Run #32009425439](https://github.com/sgl-project/sglang/actions/runs/32009425439)
- **分支**: `chunyuan/pr_dp_fix`
- **总耗时**: 131.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32009425439

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.5min | 其他 | 作业日志被截断，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681330) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 环境问题 | 健康检查发现根因作业失败，导致级联跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681426) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681520) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681529) |
| base-b-test-1-npu-a3 / run (0) | 1.2min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681536) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681580) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681593) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681817) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.4min | 其他 | 健康检查失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681872) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | PR健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681922) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325682005) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示上传artifact时未找到diffusion-failures目录，说明测试可能未产生失败文件，但无法确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681330

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业被跳过，非自身代码问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681426

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3 / run (0)）因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681520

- **base-b-test-4-npu-a3 / run (1)**: 本作业在启动前的健康检查中检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败机制跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681529

- **base-b-test-1-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681536

- **base-b-test-2-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，判定为根因作业，因此本作业（base-b-test-2-npu-a3）被快速失败跳过，并非自身执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681580

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681593

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示该作业在健康检查阶段被过滤为级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被跳过执行，属于级联失败而非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681817

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业在PR测试健康检查阶段被跳过，因为根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681872

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，导致本作业被快速失败跳过，并非本作业自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681922

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325682005

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32009425439/job/95325681464) |


## [Run #32009322315](https://github.com/sgl-project/sglang/actions/runs/32009322315)
- **分支**: `amd/wan22-fp8-pertensor-quant`
- **总耗时**: 63.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32009322315

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009322315/job/95325366232) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业正常启动并执行了上传artifact步骤，但未找到diffusion-failures文件，且无测试失败或超时信息。可能因日志截断或作业被提前终止，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009322315/job/95325366232


## [Run #32009316141](https://github.com/sgl-project/sglang/actions/runs/32009316141)
- **分支**: `jiayi/GDN`
- **总耗时**: 131.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32009316141

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.3min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371486) |
| base-b-test-8-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371679) |
| base-b-test-4-npu-a3 / run (1) | 1.3min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371693) |
| base-b-test-1-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371705) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是另一个作业失败导致级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371715) |
| base-b-test-16-npu-a3 / run (0) | 1.6min | 环境问题 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371719) |
| base-b-test-2-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371794) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因同PR中其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325372115) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325372162) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325372192) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.9min | 其他 | 健康检查失败，根因是多模态生成测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325372280) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371486

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败策略被取消，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371679

- **base-b-test-4-npu-a3 / run (1)**: health-check检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371693

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371705

- **base-b-test-4-npu-a3 / run (0)**: 该作业在启动阶段因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被取消，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371715

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371719

- **base-b-test-2-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，作为根因触发fast-fail，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371794

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示health-check检测到同PR的multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖问题而非本作业自身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325372115

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业在启动阶段被PR健康检查拦截，原因是同一次运行中的multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325372162

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，本作业被Fast-fail机制跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325372192

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因该根因失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325372280

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32009316141/job/95325371637) |


## [Run #32009315233](https://github.com/sgl-project/sglang/actions/runs/32009315233)
- **分支**: `amd/fix-vae-spatial-parallel-decode-rocm-conv2d`
- **总耗时**: 57.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32009315233

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.6min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32009315233/job/95325348425) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/checkout@v4 时返回 429 Too Many Requests，触发重试后仍可能失败，属于 GitHub 服务端限流导致的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32009315233/job/95325348425


## [Run #32008686153](https://github.com/sgl-project/sglang/actions/runs/32008686153)
- **分支**: `cleanup-environ`
- **总耗时**: 137.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32008686153

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.8min | 其他 | 作业未显示明确失败原因，仅上传失败日志文件为空。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485792) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因任务失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485890) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485932) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485933) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485936) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486011) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486023) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.6min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486205) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.5min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486286) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486316) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486329) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时未找到文件，作业可能因测试未生成失败文件而正常结束，但状态被标记为失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485792

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败任务，根因是multimodal-gen-test-1-npu-a3失败，触发fast-fail，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485890

- **base-b-test-2-npu-a3 / run (0)**: 该作业在启动前的健康检查中检测到根因失败作业multimodal-gen-test-1-npu-a3，因此被跳过并快速失败，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485932

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行PR健康检查时，检测到multimodal-gen-test-1-npu-a3作业失败（根因），因此触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485933

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤后跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323485936

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486011

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，触发fast-fail跳过本作业，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486023

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486205

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486286

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于依赖的上游作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486316

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486329

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32008686153/job/95323486056) |


## [Run #32008637548](https://github.com/sgl-project/sglang/actions/runs/32008637548)
- **分支**: `bytedance/hicache-batch-pp-completions`
- **总耗时**: 137.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32008637548

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348073) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348088) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348125) |
| base-b-test-2-npu-a3 / run (0) | 1.2min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348138) |
| base-b-test-4-npu-a3 / run (0) | 1.2min | 其他 | 健康检查发现根因任务失败，触发级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348145) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348171) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348206) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 作业因其他根因作业失败而被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348354) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348364) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348389) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.5min | 其他 | PR测试健康检查失败，根因是多模态生成测试作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348391) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures工件时未找到文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348073

- **base-b-test-1-npu-a3 / run (0)**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因健康检查快速失败机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348088

- **base-b-test-8-npu-a3 / run (0)**: 作业启动后，健康检查检测到同一PR中的multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348125

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，触发fast-fail，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348138

- **base-b-test-4-npu-a3 / run (0)**: 该作业因健康检查检测到根因任务multimodal-gen-test-1-npu-a3失败，被级联过滤并快速失败，非自身代码问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348145

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查将本作业标记为级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348171

- **base-b-test-4-npu-a3 / run (1)**: 本作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业被跳过未实际运行，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348206

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，本作业被Fast-fail机制跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348354

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348364

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348389

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348391

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32008637548/job/95323348283) |


## [Run #32008616124](https://github.com/sgl-project/sglang/actions/runs/32008616124)
- **分支**: `yangminl/trtllm-mha-ragged-split2`
- **总耗时**: 135.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32008616124

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 0.8min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267755) |
| multimodal-gen-test-1-npu-a3 | 46.0min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267782) |
| base-b-test-4-npu-a3 / run (0) | 2.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267808) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267828) |
| base-b-test-8-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267867) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267878) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267939) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查检测到其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323268151) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323268170) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.9min | 其他 | 健康检查失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323268175) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323268179) |

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267755

- **multimodal-gen-test-1-npu-a3**: 日志中未包含multimodal-gen测试的具体执行输出或错误信息，仅显示runner启动、Node版本警告及artifact上传（无文件）。无法判断失败原因，可能为作业被中断或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267782

- **base-b-test-4-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而被快速失败跳过，并非自身代码或环境问题，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267808

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，属于根因失败，因此本作业（base-b-test-2-npu-a3）被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267828

- **base-b-test-8-npu-a3 / run (0)**: 该作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267867

- **base-b-test-1-npu-a3 / run (0)**: 该作业在启动时执行健康检查，发现multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267878

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败被取消，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267939

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被过滤并跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323268151

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动前的健康检查阶段检测到另一个作业multimodal-gen-test-1-npu-a3失败，触发了fast-fail机制，本作业被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323268170

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业因PR健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323268175

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323268179

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32008616124/job/95323267803) |


## [Run #32008339065](https://github.com/sgl-project/sglang/actions/runs/32008339065)
- **分支**: `cjc/unified-swa-branching`
- **总耗时**: 22.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32008339065

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 2.1min | 其他 | 作业日志被截断，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491303) |
| base-b-test-16-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491391) |
| base-b-test-1-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491443) |
| base-b-test-4-npu-a3 / run (1) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491462) |
| base-b-test-4-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491466) |
| base-b-test-8-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491489) |
| base-a-test-1-npu-a2 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491550) |
| base-b-test-2-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491553) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491802) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491846) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491920) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491962) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行的具体输出。仅能看到上传diffusion-failures工件时提示无文件，说明测试可能未产生失败样本，但无法确认测试是否通过或失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491303

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误所致，属于基础设施或环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491391

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491443

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491462

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491466

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491489

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491550

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491553

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491802

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491846

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491920

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或被误删，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32008339065/job/95322491962


## [Run #32007903752](https://github.com/sgl-project/sglang/actions/runs/32007903752)
- **分支**: `new_epd`
- **总耗时**: 145.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007903752

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 49.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189375) |
| base-b-test-1-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189466) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189511) |
| base-b-test-16-npu-a3 / run (0) | 3.0min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189515) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189522) |
| base-b-test-4-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189551) |
| base-b-test-8-npu-a3 / run (0) | 0.7min | 环境问题 | 健康检查发现其他作业根因失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189606) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查失败，根因是另一个作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189749) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189771) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189791) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189871) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或失败原因被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189375

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤并跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189466

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动前的健康检查中检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189511

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189515

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189522

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189551

- **base-b-test-8-npu-a3 / run (0)**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因健康检查失败被快速失败机制跳过，属于级联失败，非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189606

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业因PR健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而被快速跳过，并非自身测试失败，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189749

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189771

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189791

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业在启动阶段因PR健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189871

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007903752/job/95321189490) |


## [Run #32007863939](https://github.com/sgl-project/sglang/actions/runs/32007863939)
- **分支**: `consensus_checker_0806`
- **总耗时**: 143.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007863939

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 51.2min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159649) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159685) |
| base-b-test-8-npu-a3 / run (0) | 2.5min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159746) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159805) |
| base-b-test-16-npu-a3 / run (0) | 1.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159945) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159984) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160050) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160405) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.6min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160433) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160445) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.2min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160528) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node.js版本弃用警告和上传artifact时无文件提示，无法判断具体失败原因，可能为作业被中断或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159649

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159685

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159746

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业被跳过未实际运行，属于依赖作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159805

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败被取消，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159945

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3 run 0）因级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159984

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败策略被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160050

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160405

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160433

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail跳过本作业，并非本作业自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160445

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321160528

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007863939/job/95321159750) |


## [Run #32007855113](https://github.com/sgl-project/sglang/actions/runs/32007855113)
- **分支**: `pllimax/output-log-dir-structure`
- **总耗时**: 17.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007855113

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082501) |
| base-b-test-16-npu-a3 / run (0) | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082596) |
| base-b-test-2-npu-a3 / run (0) | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082629) |
| base-a-test-1-npu-a2 / run (0) | 17.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082639) |
| base-b-test-4-npu-a3 / run (1) | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082640) |
| base-b-test-4-npu-a3 / run (0) | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082653) |
| base-b-test-8-npu-a3 / run (0) | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082823) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082908) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082924) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082981) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321083093) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082501

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082596

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082629

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082639

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082640

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）未上传或已被删除，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082653

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082823

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082908

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082924

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321082981

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007855113/job/95321083093


## [Run #32007823930](https://github.com/sgl-project/sglang/actions/runs/32007823930)
- **分支**: `hybrid_tree/change-default`
- **总耗时**: 140.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007823930

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 53.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039848) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039889) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039936) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039981) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039983) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因作业失败，本作业被级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040000) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040039) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.0min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040332) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.0min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040384) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.9min | 其他 | PR健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040431) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040439) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅显示上传diffusion-failures工件时未找到文件，可能测试未运行或日志被截断，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039848

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤，最终因快速失败机制终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039889

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039936

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039981

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3）因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039983

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040000

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040039

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040332

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040384

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被过滤，实际未执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040431

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示本作业在“Check PR test health”步骤因检测到根因作业multimodal-gen-test-1-npu-a3失败而触发fast-fail，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321040439

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007823930/job/95321039965) |


## [Run #32007820836](https://github.com/sgl-project/sglang/actions/runs/32007820836)
- **分支**: `dsv4_fp8_trtllm_gen`
- **总耗时**: 141.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007820836

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 46.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054431) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054501) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054506) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054549) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054569) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054585) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054705) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054749) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054757) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.3min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054774) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054901) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或失败原因被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054431

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054501

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败策略被取消，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054506

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业（base-b-test-4-npu-a3）被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054549

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054569

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054585

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054705

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 本作业在启动前的PR健康检查阶段被快速失败（fast-fail），原因是同一次运行中的根因作业multimodal-gen-test-1-npu-a3失败，本作业并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054749

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段被健康检查机制拦截，原因是同一次运行中multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail逻辑，本作业未实际执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054757

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动前的PR健康检查中，检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054774

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054901

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007820836/job/95321054565) |


## [Run #32007532052](https://github.com/sgl-project/sglang/actions/runs/32007532052)
- **分支**: `wca-rebased`
- **总耗时**: 142.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007532052

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 2.2min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084375) |
| multimodal-gen-test-1-npu-a3 | 55.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084427) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084428) |
| base-b-test-8-npu-a3 / run (0) | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084473) |
| base-b-test-4-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084481) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084494) |
| base-b-test-1-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084520) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320085207) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320085345) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 0.9min | 其他 | PR健康检查失败，根因是另一个作业multimodal-gen-test-1-npu-a3失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320085351) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320085364) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，触发了fast-fail，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084375

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但无法从日志判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084427

- **base-b-test-2-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084428

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084473

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤，最终因快速失败策略退出，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084481

- **base-b-test-4-npu-a3 / run (1)**: 健康检查显示multimodal-gen-test-1-npu-a3为根因失败，本作业因级联失败被过滤，随后快速失败退出，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084494

- **base-b-test-1-npu-a3 / run (0)**: 该作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，根据快速失败策略主动终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084520

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业因级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320085207

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查过滤掉多个级联失败后，根因失败作业为multimodal-gen-test-1-npu-a3，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320085345

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 本作业在启动前的PR测试健康检查阶段被快速失败机制跳过，根因是multimodal-gen-test-1-npu-a3作业失败，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320085351

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320085364

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007532052/job/95320084580) |


## [Run #32007504074](https://github.com/sgl-project/sglang/actions/runs/32007504074)
- **分支**: `amd-mla-decode-gfx950-tune`
- **总耗时**: 142.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007504074

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 51.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080096) |
| base-b-test-16-npu-a3 / run (0) | 2.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080224) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080267) |
| base-b-test-2-npu-a3 / run (0) | 2.0min | 环境问题 | 健康检查检测到根因作业失败，触发级联失败快速终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080315) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080342) |
| base-b-test-1-npu-a3 / run (0) | 0.6min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080380) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080430) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080725) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080829) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080835) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 0.9min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080880) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions环境准备、Node版本警告及上传失败产物（无文件）的提示，无法判断测试失败的具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080096

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080224

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3为根因失败，本作业因级联失败被过滤后仍被快速失败跳过，非自身代码问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080267

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业后，识别出根因作业为multimodal-gen-test-1-npu-a3，导致本作业被快速失败跳过，属于环境或依赖作业问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080315

- **base-b-test-4-npu-a3 / run (1)**: 本作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080342

- **base-b-test-1-npu-a3 / run (0)**: 健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080380

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080430

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤并跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080725

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080829

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080835

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080880

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007504074/job/95320080353) |


## [Run #32007288787](https://github.com/sgl-project/sglang/actions/runs/32007288787)
- **分支**: `feat/triton-sparse-mla`
- **总耗时**: 142.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007288787

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 45.6min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442354) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442479) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442488) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442494) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442528) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442591) |
| base-b-test-16-npu-a3 / run (0) | 2.2min | 其他 | 健康检查快速失败，根因作业为multimodal-gen-test-1-npu-a3 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442657) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442848) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442850) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442856) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442858) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node.js弃用警告及上传artifact步骤，未包含任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442354

- **base-b-test-1-npu-a3 / run (0)**: 该作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，作为根因作业触发了快速失败（fast-fail），导致本作业被跳过执行，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442479

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442488

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442494

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被Fast-fail机制跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442528

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动前的健康检查中检测到根因作业multimodal-gen-test-1-npu-a3失败，根据快速失败策略，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442591

- **base-b-test-16-npu-a3 / run (0)**: 该作业本身未执行测试，因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而触发快速失败机制，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442657

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败策略被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442848

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442850

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442856

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442858

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007288787/job/95319442570) |


## [Run #32007270370](https://github.com/sgl-project/sglang/actions/runs/32007270370)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 140.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007270370

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383810) |
| base-b-test-16-npu-a3 / run (0) | 2.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383861) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383913) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383925) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383932) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383956) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383976) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.0min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319384188) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319384245) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319384274) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.8min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319384329) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本或测试未执行到该阶段，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383810

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383861

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-8-npu-a3）被快速失败跳过，并非自身执行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383913

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383925

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到同一PR的另一个作业multimodal-gen-test-1-npu-a3失败，触发了fast-fail机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383932

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，属于根因失败，因此本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383956

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-1-npu-a3）被快速失败机制跳过，并非自身执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383976

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319384188

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319384245

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，导致本作业在启动前被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319384274

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因该根因失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319384329

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007270370/job/95319383871) |


## [Run #32007034012](https://github.com/sgl-project/sglang/actions/runs/32007034012)
- **分支**: `fix/xpu-decode-graph-runner-is-current-stream-capturing`
- **总耗时**: 143.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32007034012

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 47.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传artifact时无失败文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621391) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621514) |
| base-b-test-8-npu-a3 / run (0) | 5.5min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621616) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621626) |
| base-b-test-16-npu-a3 / run (0) | 4.5min | 其他 | 健康检查快速失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621649) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621769) |
| base-b-test-4-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621783) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 作业因健康检查检测到其他根因作业失败而被快速失败跳过，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621870) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621889) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.6min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318622004) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318622090) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最终上传diffusion-failures目录时提示无文件，说明测试可能通过或失败信息未收集，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621391

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621514

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621616

- **base-b-test-2-npu-a3 / run (0)**: 该作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，属于根因失败，因此本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621626

- **base-b-test-16-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非自身测试失败，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621649

- **base-b-test-4-npu-a3 / run (1)**: 该作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业被跳过未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621769

- **base-b-test-4-npu-a3 / run (0)**: health-check检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被Fast-fail跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621783

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，导致本作业被Fast-fail跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621870

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621889

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动前的健康检查阶段检测到同一PR中multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业被跳过未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318622004

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318622090

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32007034012/job/95318621598) |


## [Run #32006924715](https://github.com/sgl-project/sglang/actions/runs/32006924715)
- **分支**: `ling3-flash-dspark`
- **总耗时**: 140.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32006924715

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.2min | 其他 | 作业未显示明确失败原因，可能为测试通过或日志截断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364758) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364896) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364900) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364904) |
| base-b-test-1-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364918) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364956) |
| base-b-test-8-npu-a3 / run (0) | 0.7min | 其他 | 健康检查检测到其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365011) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365214) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365266) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365279) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.3min | 其他 | 健康检查失败，根因作业为multimodal-gen-test-1-npu-a3，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365406) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含环境准备、Node 20弃用警告及上传artifact时未找到diffusion-failures目录，未出现测试失败或错误信息，可能测试实际通过或日志不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364758

- **base-b-test-4-npu-a3 / run (0)**: 作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，按策略快速失败，本作业未实际执行测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364896

- **base-b-test-4-npu-a3 / run (1)**: 该作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业被跳过未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364900

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364904

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364918

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364956

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤，实际未执行测试，属于CI流程的快速失败保护机制。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365011

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，导致本作业（base-c-test-acc-4-npu-a3）在启动前被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365214

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365266

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业因快速失败（fast-fail）被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365279

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败机制跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318365406

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32006924715/job/95318364939) |


## [Run #32006905491](https://github.com/sgl-project/sglang/actions/runs/32006905491)
- **分支**: `pllimax/output-log-dir-structure`
- **总耗时**: 13.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32006905491

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 12.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234116) |
| base-b-test-1-npu-a3 / run (0) | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234140) |
| base-b-test-4-npu-a3 / run (0) | 12.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234153) |
| base-b-test-2-npu-a3 / run (0) | 12.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234183) |
| base-b-test-4-npu-a3 / run (1) | 12.7min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234198) |
| base-b-test-8-npu-a3 / run (0) | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234204) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234319) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234447) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234531) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234567) |

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234116

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234140

- **base-b-test-4-npu-a3 / run (0)**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误。这通常是因为文件被删除、路径错误或存储账户配置问题，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234153

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234183

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234198

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234204

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234319

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234447

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234531

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234567

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32006905491/job/95318234082) |


## [Run #32006713528](https://github.com/sgl-project/sglang/actions/runs/32006713528)
- **分支**: `codex/diffusion-reuse-srt-clip`
- **总耗时**: 138.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32006713528

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 64.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317665897) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317665982) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666056) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666071) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666081) |
| base-b-test-16-npu-a3 / run (0) | 1.8min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666088) |
| base-b-test-8-npu-a3 / run (0) | 1.0min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666143) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666247) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666342) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.0min | 其他 | 健康检查失败，根因是多模态生成测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666348) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666401) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传artifact时未找到diffusion-failures目录，说明测试可能未产生失败文件，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317665897

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被取消，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317665982

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被Fast-fail机制跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666056

- **base-b-test-4-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因作业，因此本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666071

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3）被级联过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666081

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666088

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666143

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666247

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666342

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666348

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查过滤后根因失败作业为multimodal-gen-test-1-npu-a3，本作业因Fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666401

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32006713528/job/95317666035) |


## [Run #32006710139](https://github.com/sgl-project/sglang/actions/runs/32006710139)
- **分支**: `new_epd`
- **总耗时**: 10.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32006710139

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683522) |
| base-b-test-1-npu-a3 / run (0) | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683588) |
| base-a-test-1-npu-a2 / run (0) | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683629) |
| base-b-test-4-npu-a3 / run (1) | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683686) |
| base-b-test-4-npu-a3 / run (0) | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683712) |
| base-b-test-2-npu-a3 / run (0) | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683724) |
| base-b-test-8-npu-a3 / run (0) | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683756) |
| base-b-test-16-npu-a3 / run (0) | 9.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683795) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317684383) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317684425) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317684450) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317684539) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683522

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683588

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683629

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683686

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683712

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683724

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683756

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317683795

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317684383

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317684425

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317684450

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32006710139/job/95317684539


---
*Auto-generated by npu_pr_monitor.py*