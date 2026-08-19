# NPU CI 执行监控
**生成时间**: 2026-08-19 08:50 UTC
**分析 Run 数**: 54

---

## 📊 本次执行总结

- **成功 Job 数**: 131
- **失败 Run 数**: 54
- **成功 Job 平均耗时**: 25.3min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.8min | #32207210932 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707814) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.7min | #32206508270 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930779097) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 102.5min | #32210758224 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908476) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 97.3min | #32206889361 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833957) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 91.5min | #32210047073 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852112) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.8min | #32210674597 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352761) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.8min | #32205599687 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269295) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.2min | #32209218935 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433757) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.1min | #32212182766 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946964190) |
| base-b-test-16-npu-a3 / run (0) | 83.7min | #32212182766 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963987) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 52 | 0 | 33 | 0 | 19 |
| base-b-test-16-npu-a3 / run (0) | 45 | 4 | 1 | 9 | 31 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 9 | 0 | 0 | 9 | 0 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 45 | 6 | 0 | 8 | 31 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 10 | 3 | 0 | 7 | 0 |
| base-b-test-8-npu-a3 / run (0) | 45 | 8 | 0 | 6 | 31 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 45 | 8 | 0 | 6 | 31 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 6 | 0 | 0 | 6 | 0 |
| base-b-test-2-npu-a3 / run (0) | 45 | 9 | 0 | 5 | 31 |
| base-a-test-1-npu-a2 / run (0) | 45 | 39 | 0 | 5 | 1 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 45 | 9 | 0 | 5 | 31 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8 | 3 | 0 | 5 | 0 |
| base-b-test-1-npu-a3 / run (0) | 45 | 10 | 0 | 4 | 31 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 45 | 10 | 0 | 4 | 31 |
| base-b-test-4-npu-a3 / run (0) | 45 | 11 | 0 | 3 | 31 |
| base-b-test-4-npu-a3 / run (1) | 45 | 11 | 0 | 3 | 31 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32207210932<br>[#32611 Fix transcription & audio-understanding for ASR/audio/speech models](https://github.com/sgl-project/sglang/pull/32611) | `enable_audio_model_transcription` | 230.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32207210932) |
| #32206508270<br>[#24959 XPU: Enable GLM5.1 (GlmMoeDsaForCausalLM) DSA Attention](https://github.com/sgl-project/sglang/pull/24959) | `glm5.1_enabling` | 228.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32206508270) |
| #32206889361<br>[#34936 [NPU] [FIX] Fix non-contiguous parameter issue in FIA operator](https://github.com/sgl-project/sglang/pull/34936) | `br_krope_conti` | 225.5min | 2 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32206889361) |
| #32212182766<br>[#34198 perf(kimi-k3): fuse ROCm KDA decode boundary](https://github.com/sgl-project/sglang/pull/34198) | `perf/k3_fused_kda_decode` | 219.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32212182766) |
| #32205599687<br>[#35401 [Fix] Write the req_to_token page tail so rows stay valid over whole pages](https://github.com/sgl-project/sglang/pull/35401) | `lsyin/page-tail-write` | 202.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32205599687) |
| #32212629941<br>[#35412 [Fix] Land the decode mamba checkpoint depth on the tree page under DCP](https://github.com/sgl-project/sglang/pull/35412) | `kpham/mamba-track-interval-tree-page` | 201.6min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32212629941) |
| #32210758224<br>[#31689 [Kernel] Avoid batch-size specialization in masked KV writes](https://github.com/sgl-project/sglang/pull/31689) | `codex/fix-masked-kv-constexpr` | 195.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32210758224) |
| #32210674597<br>[#35435 [sgl-kernel][CPU] Add group-aware CPU SHM collective kernels](https://github.com/sgl-project/sglang/pull/35435) | `cpu-shm-process-group` | 190.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32210674597) |
| #32209218935<br>[#33863 [Feature] PP Support PD + DSpark](https://github.com/sgl-project/sglang/pull/33863) | `deepseek_v4_dspark_suppport_pp_pd` | 189.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32209218935) |
| #32210047073<br>[#35269 [UnifiedTree] feat: support runtime attach/detach](https://github.com/sgl-project/sglang/pull/35269) | `feature/unified-runtime-attach` | 185.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32210047073) |
| #32205927495 | `cursor/fix-multimodal-gen-1gpu-amd-rocm-d1f1` | 137.6min | 0 |  | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32205927495) |
| #32207835665<br>[#35304 [XPU][model]: Intel XPU support for encoder embeddings (bge/NomicBERT/ModernBERT) + InternVL3_5](https://github.com/sgl-project/sglang/pull/35304) | `model-serve/encoder-internvl-xpu` | 136.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32207835665) |
| #32207382259<br>[#35349 [VLM] Default to two multimodal preprocessing workers](https://github.com/sgl-project/sglang/pull/35349) | `claude/mm-processor-concurrency-default` | 126.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32207382259) |
| #32206882148<br>[#32856 [CPU] Fix NUMA/core binding for DP ranks](https://github.com/sgl-project/sglang/pull/32856) | `chunyuan/pr_dp_fix` | 119.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32206882148) |
| #32207387118<br>[#33323 [Intel XPU] Add xpu pass for biased_topk and hash_topk](https://github.com/sgl-project/sglang/pull/33323) | `gaopengf/enable_more_topk_for_xpu` | 97.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32207387118) |
| #32212672686 | `fix-kimi-k3-attn-res-tma-sm120-gate` | 94.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32212672686) |
| #32224436675<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 76.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32224436675) |
| #32214123419<br>[#35360 [PD] Deferred decode-side KV release for the NIXL backend](https://github.com/sgl-project/sglang/pull/35360) | `feat/nixl-deferred-decode-kv-release` | 75.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32214123419) |
| #32208402261<br>[#35428 [AMD] Fix gfx950 Triton compiler crash on fp8 KV-cache attention](https://github.com/sgl-project/sglang/pull/35428) | `cursor/fix-gfx950-fp8-kv-triton-attention-crash-4ddf` | 72.7min | 10 | base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32208402261) |
| #32212949557<br>[#35308 [AMD] Fuse Kimi-K3 MLA Q and cache preparation](https://github.com/sgl-project/sglang/pull/35308) | `perf/k3-mla-q-cache-fusion` | 60.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32212949557) |
| #32216276120<br>[#35215 [Constrained] Support MistralCommon tokenizers in the XGrammar backend](https://github.com/sgl-project/sglang/pull/35215) | `main` | 60.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32216276120) |
| #32213277921<br>[#33838 [AMD] Perf Kimi-K3 MoE optimization](https://github.com/sgl-project/sglang/pull/33838) | `perf/k3_moe-opt` | 55.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32213277921) |
| #32221100887<br>[#35349 [VLM] Default to two multimodal preprocessing workers](https://github.com/sgl-project/sglang/pull/35349) | `claude/mm-processor-concurrency-default` | 54.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32221100887) |
| #32212503618<br>[#35306 [mem_cache][9/N] refactor: move DSAIndexerPoolHost to pool_host.dsa](https://github.com/sgl-project/sglang/pull/35306) | `refactor/mem-cache-poolhost-dsa` | 54.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32212503618) |
| #32208109042<br>[#33880 [diffusion] optimization: reduce minimax h3 mps memory pressure](https://github.com/sgl-project/sglang/pull/33880) | `codex/minimax-h3-mps` | 52.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32208109042) |
| #32211161960<br>[#35298 [Fix] DCP: advertise the logical KV-event block size](https://github.com/sgl-project/sglang/pull/35298) | `fix-dcp-args-validation` | 45.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32211161960) |
| #32209504383 | `fix-kimi-k3-attn-res-tma-sm120-gate` | 43.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32209504383) |
| #32224368829<br>[#35336 VLM: feed the packed qkv projection output to vision backends uncopied](https://github.com/sgl-project/sglang/pull/35336) | `main` | 38.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32224368829) |
| #32209930978<br>[#34198 perf(kimi-k3): fuse ROCm KDA decode boundary](https://github.com/sgl-project/sglang/pull/34198) | `perf/k3_fused_kda_decode` | 37.8min | 11 | base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32209930978) |
| #32213237737<br>[#34485 [AMD] Let the diffusion AITer backend take grouped-query K/V (fix Cosmos3-Nano startup)](https://github.com/sgl-project/sglang/pull/34485) | `cursor/amd-aiter-diffusion-gqa-d1f1` | 37.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32213237737) |
| #32212644786<br>[#35287 [AMD] Add independent Kimi-K3 gfx950 FlyDSL integrations](https://github.com/sgl-project/sglang/pull/35287) | `perf/k3-gfx950-independent-fusions-clean` | 35.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32212644786) |
| #32210194241<br>[#34953 [Perf] Restore the 16-token router GEMM threshold on SM10X](https://github.com/sgl-project/sglang/pull/34953) | `main` | 33.7min | 11 | base-b-test-8-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32210194241) |
| #32207601263<br>[#35174 [Diffusion] Reuse shared checkpoint quant metadata resolver](https://github.com/sgl-project/sglang/pull/35174) | `codex/diffusion-use-checkpoint-quant-spec` | 32.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32207601263) |
| #32213855034<br>[#35298 [Fix] DCP: advertise the logical KV-event block size](https://github.com/sgl-project/sglang/pull/35298) | `main` | 30.5min | 11 | base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32213855034) |
| #32212209374<br>[#33473 [HiCache] Batch PP write and load completion sync](https://github.com/sgl-project/sglang/pull/33473) | `main` | 27.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32212209374) |
| #32224262745<br>[#35456 [AMD] [CI] Fix ROCm aiter attention backend rejecting grouped-query attention](https://github.com/sgl-project/sglang/pull/35456) | `amd/fix-aiter-attention-gqa` | 25.6min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32224262745) |
| #32208880554<br>[#34680 [diffusion][Minimax H3]support subblock sparse attention on SM90](https://github.com/sgl-project/sglang/pull/34680) | `main` | 22.2min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32208880554) |
| #32221181939<br>[#35352 [diffusion] ComfyUI: add a MiniMax-H3 node and a generic extra-fields passthrough](https://github.com/sgl-project/sglang/pull/35352) | `codex/comfyui-minimax-h3-node` | 21.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32221181939) |
| #32225449998<br>[#35412 [Fix] Land the decode mamba checkpoint depth on the tree page under DCP](https://github.com/sgl-project/sglang/pull/35412) | `kpham/mamba-track-interval-tree-page` | 20.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32225449998) |
| #32227160698<br>[#34485 [AMD] Let the diffusion AITer backend take grouped-query K/V (fix Cosmos3-Nano startup)](https://github.com/sgl-project/sglang/pull/34485) | `main` | 20.2min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32227160698) |
| #32223077541<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 20.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32223077541) |
| #32206737801<br>[#12961 Fix DP attention on CPU](https://github.com/sgl-project/sglang/pull/12961) | `main` | 16.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32206737801) |
| #32220500767<br>[#35183 refactor(diffusion): gate native encoder quantized checkpoints](https://github.com/sgl-project/sglang/pull/35183) | `codex/native-encoder-quant-capability` | 16.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32220500767) |
| #32205810757<br>[#23112 Add fmha_v2 attention backend for SM90/120](https://github.com/sgl-project/sglang/pull/23112) | `main` | 15.5min | 11 | base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32205810757) |
| #32220091736<br>[#35174 [Diffusion] Reuse shared checkpoint quant metadata resolver](https://github.com/sgl-project/sglang/pull/35174) | `main` | 13.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32220091736) |
| #32221739127<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 13.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32221739127) |
| #32228690154<br>[#35424 [Fix] Scale the req_to_token row headroom by attn_dcp_size](https://github.com/sgl-project/sglang/pull/35424) | `main` | 11.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32228690154) |
| #32225321161<br>[#35469 Warm PaddleOCR-VL with a page-sized image, not a 32x32 one](https://github.com/sgl-project/sglang/pull/35469) | `claude/vlm-warmup-image-size` | 8.9min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32225321161) |
| #32223077472 | `fix-kimi-k3-attn-res-tma-sm120-gate` | 8.9min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32223077472) |
| #32207742803<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `main` | 8.8min | 11 | base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32207742803) |
| #32208261134<br>[#34798 [HiCache] Buffer-only mode for HiCache host memory layer](https://github.com/sgl-project/sglang/pull/34798) | `main` | 7.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32208261134) |
| #32226982379<br>[#34424 [AMD] Fix ROCm VAE Conv2D fast path breaking spatial-parallel decode](https://github.com/sgl-project/sglang/pull/34424) | `amd/fix-vae-spatial-parallel-decode-rocm-conv2d` | 7.4min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32226982379) |
| #32222668777<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 6.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32222668777) |
| #32205486969<br>[#23112 Add fmha_v2 attention backend for SM90/120](https://github.com/sgl-project/sglang/pull/23112) | `fmha_v2` | 6.0min | 11 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-b-test-8-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32205486969) |

---


## [Run #32228690154](https://github.com/sgl-project/sglang/actions/runs/32228690154)
- **分支**: `main`
- **总耗时**: 11.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32228690154

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877413) |
| base-b-test-8-npu-a3 / run (0) | 9.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877470) |
| base-b-test-2-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877525) |
| base-b-test-4-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877535) |
| base-b-test-4-npu-a3 / run (1) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877598) |
| base-b-test-1-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877624) |
| base-b-test-16-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877655) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877880) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877902) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877907) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877959) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含runner初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现测试执行或失败断言，无法判断具体失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877413

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877470

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是上游作业未成功上传或存储配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877525

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877535

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877598

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877624

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877655

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877880

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877902

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877907

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877959

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32228690154/job/95993877614) |


## [Run #32227160698](https://github.com/sgl-project/sglang/actions/runs/32227160698)
- **分支**: `main`
- **总耗时**: 20.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32227160698

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32227160698/job/95989277318) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未出现测试执行或失败的具体错误，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32227160698/job/95989277318


## [Run #32226982379](https://github.com/sgl-project/sglang/actions/runs/32226982379)
- **分支**: `amd/fix-vae-spatial-parallel-decode-rocm-conv2d`
- **总耗时**: 7.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32226982379

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.7min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32226982379/job/95988646479) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32226982379/job/95988646479


## [Run #32225449998](https://github.com/sgl-project/sglang/actions/runs/32225449998)
- **分支**: `kpham/mamba-track-interval-tree-page`
- **总耗时**: 20.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32225449998

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406160) |
| base-b-test-4-npu-a3 / run (0) | 19.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406243) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406262) |
| base-b-test-16-npu-a3 / run (0) | 19.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406284) |
| base-b-test-4-npu-a3 / run (1) | 19.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406353) |
| base-b-test-8-npu-a3 / run (0) | 19.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406380) |
| base-b-test-1-npu-a3 / run (0) | 19.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406413) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406553) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406704) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406720) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.0min | 环境问题 | CI 依赖的 Azure Blob 存储文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406722) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406160

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406243

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传失败、路径错误或存储被清理所致，属于基础设施或配置问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406262

- **base-b-test-16-npu-a3 / run (0)**: 作业失败原因是BlobNotFound错误，即请求的blob在存储中不存在，可能是资源被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406284

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，请求的资源在存储中不存在，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406353

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406380

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406413

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406553

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406704

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406720

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明作业尝试下载的 blob 资源缺失或路径错误，可能是上游产物未上传或存储被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406722

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32225449998/job/95984406240) |


## [Run #32225321161](https://github.com/sgl-project/sglang/actions/runs/32225321161)
- **分支**: `claude/vlm-warmup-image-size`
- **总耗时**: 8.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32225321161

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983843899) |
| base-a-test-1-npu-a2 / run (0) | 7.8min | 环境问题 | 自定义容器执行失败，NPU测试环境未正常启动 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983843946) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983843956) |
| base-b-test-2-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844014) |
| base-b-test-1-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844035) |
| base-b-test-16-npu-a3 / run (0) | 8.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844047) |
| base-b-test-8-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844053) |
| base-b-test-4-npu-a3 / run (1) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844136) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844242) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844304) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844325) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844692) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983843899

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行run_suite.py前，执行自定义容器实现时失败，报错'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983843946

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983843956

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问 Azure Blob 中的某个 blob，但该 blob 不存在（BlobNotFound）。这通常是因为日志文件未上传、路径错误或已被删除，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844014

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844035

- **base-b-test-16-npu-a3 / run (0)**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844047

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查 blob 名称和存储账户状态。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844053

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844136

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844242

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844304

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844325

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32225321161/job/95983844692


## [Run #32224436675](https://github.com/sgl-project/sglang/actions/runs/32224436675)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 76.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32224436675

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 68.8min | 其他 | 作业日志不完整，缺少实际测试执行和失败信息，仅显示上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438728) |
| base-b-test-16-npu-a3 / run (0) | 74.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438881) |
| base-b-test-4-npu-a3 / run (0) | 74.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438923) |
| base-b-test-1-npu-a3 / run (0) | 74.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438944) |
| base-b-test-8-npu-a3 / run (0) | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439020) |
| base-b-test-4-npu-a3 / run (1) | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439034) |
| base-b-test-2-npu-a3 / run (0) | 74.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439105) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 74.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439338) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 74.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439405) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 74.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439408) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 74.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439409) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试运行的具体输出或错误，仅显示上传diffusion-failures目录时未找到文件，可能测试未执行或提前退出，需查看完整日志定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438728

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438881

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438923

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438944

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、缓存或日志）在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查 CI 配置中的 blob 路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439020

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439034

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439105

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439338

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439405

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439408

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981439409

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32224436675/job/95981438999) |


## [Run #32224368829](https://github.com/sgl-project/sglang/actions/runs/32224368829)
- **分支**: `main`
- **总耗时**: 38.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32224368829

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 34.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049391) |
| base-b-test-2-npu-a3 / run (0) | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049482) |
| base-b-test-1-npu-a3 / run (0) | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049486) |
| base-b-test-16-npu-a3 / run (0) | 37.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049489) |
| base-b-test-4-npu-a3 / run (0) | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049525) |
| base-b-test-8-npu-a3 / run (0) | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049531) |
| base-b-test-4-npu-a3 / run (1) | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049562) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049760) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049774) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049857) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049926) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions运行环境准备、Node.js弃用警告及上传artifact时未找到文件的提示，无法判断测试失败的具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049391

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049482

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049486

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049489

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049525

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049531

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049562

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049760

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049774

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049857

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049926

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32224368829/job/95981049451) |


## [Run #32224262745](https://github.com/sgl-project/sglang/actions/runs/32224262745)
- **分支**: `amd/fix-aiter-attention-gqa`
- **总耗时**: 25.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32224262745

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 20.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32224262745/job/95980745666) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含多模态生成测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32224262745/job/95980745666


## [Run #32223077541](https://github.com/sgl-project/sglang/actions/runs/32223077541)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 20.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32223077541

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357649) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357791) |
| base-b-test-4-npu-a3 / run (0) | 18.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357809) |
| base-b-test-4-npu-a3 / run (1) | 18.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357834) |
| base-b-test-8-npu-a3 / run (0) | 18.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357836) |
| base-b-test-16-npu-a3 / run (0) | 18.7min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357850) |
| base-b-test-1-npu-a3 / run (0) | 18.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357869) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 18.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977358072) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 18.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977358093) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 18.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977358108) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 18.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977358117) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357649

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357791

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357809

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是上传失败或路径错误，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357834

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357836

- **base-b-test-16-npu-a3 / run (0)**: 作业失败是因为访问的Blob资源返回404错误（BlobNotFound），可能是CI流程中上传或引用工件路径有误，或存储容器被清理。需检查作业依赖的工件是否已正确生成并上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357850

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357869

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977358072

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977358093

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到，可能是 CI 依赖的构建产物或缓存被清理或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977358108

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977358117

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077541/job/95977357805) |


## [Run #32223077472](https://github.com/sgl-project/sglang/actions/runs/32223077472)
- **分支**: `fix-kimi-k3-attn-res-tma-sm120-gate`
- **总耗时**: 8.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32223077472

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977337885) |
| base-b-test-1-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338093) |
| base-b-test-16-npu-a3 / run (0) | 8.2min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338095) |
| base-b-test-8-npu-a3 / run (0) | 8.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338137) |
| base-b-test-4-npu-a3 / run (1) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338261) |
| base-b-test-2-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338274) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338330) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338449) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338480) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338484) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338666) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977337885

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338093

- **base-b-test-16-npu-a3 / run (0)**: 作业日志显示BlobNotFound错误，说明CI流程尝试下载的blob（可能为测试数据或构建产物）已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338095

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338137

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338261

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338274

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是上传失败或路径错误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338330

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失，可能是资源被清理或路径配置错误，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338449

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338480

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径错误，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338484

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338666

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32223077472/job/95977338161) |


## [Run #32222668777](https://github.com/sgl-project/sglang/actions/runs/32222668777)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 6.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32222668777

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138206) |
| base-b-test-1-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138311) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | 环境问题 | 自定义容器执行失败，NPU测试无法启动 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138356) |
| base-b-test-2-npu-a3 / run (0) | 5.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138368) |
| base-b-test-4-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138411) |
| base-b-test-4-npu-a3 / run (1) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138465) |
| base-b-test-8-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138473) |
| base-b-test-16-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138487) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138784) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138856) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138890) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138914) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138206

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置不一致，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138311

- **base-a-test-1-npu-a2 / run (0)**: 日志显示测试开始后立即报错"Executing the custom container implementation failed"，说明NPU容器环境初始化失败，导致测试未实际运行即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138356

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中已被删除或路径错误，属于基础设施或配置问题，需检查CI脚本中的资源引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138368

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138411

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure 存储中缺失或已被删除，可能是上传失败或路径错误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138465

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储中的文件不存在，可能是资源被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138473

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问的 blob 资源未找到（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138487

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138784

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是资源清理或路径配置错误，需检查相关存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138856

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138890

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32222668777/job/95976138914


## [Run #32221739127](https://github.com/sgl-project/sglang/actions/runs/32221739127)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 13.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32221739127

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417783) |
| base-b-test-8-npu-a3 / run (0) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417833) |
| base-b-test-2-npu-a3 / run (0) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417886) |
| base-b-test-1-npu-a3 / run (0) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417909) |
| base-b-test-4-npu-a3 / run (0) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417921) |
| base-b-test-4-npu-a3 / run (1) | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417959) |
| base-b-test-16-npu-a3 / run (0) | 13.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418010) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418165) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418185) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418201) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418264) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417783

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417833

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417886

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417909

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个 Azure 存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417921

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417959

- **base-b-test-16-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，表明CI流程尝试访问的存储对象缺失或路径错误，可能是上传失败、清理策略或配置问题，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418010

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418165

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418185

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已丢失或路径错误，属于外部存储环境问题，需检查相关 blob 是否存在或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418201

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置变更所致，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973418264

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32221739127/job/95973417826) |


## [Run #32221181939](https://github.com/sgl-project/sglang/actions/runs/32221181939)
- **分支**: `codex/comfyui-minimax-h3-node`
- **总耗时**: 21.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32221181939

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221181939/job/95971893155) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221181939/job/95971893155


## [Run #32221100887](https://github.com/sgl-project/sglang/actions/runs/32221100887)
- **分支**: `claude/mm-processor-concurrency-default`
- **总耗时**: 54.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32221100887

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 35.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715443) |
| base-b-test-2-npu-a3 / run (0) | 53.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715515) |
| base-b-test-1-npu-a3 / run (0) | 53.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715551) |
| base-b-test-4-npu-a3 / run (1) | 53.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715580) |
| base-b-test-4-npu-a3 / run (0) | 53.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715587) |
| base-b-test-16-npu-a3 / run (0) | 53.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715626) |
| base-b-test-8-npu-a3 / run (0) | 53.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715673) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 53.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971716062) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 53.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971716176) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 53.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971716185) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 53.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971716511) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的输出，仅有GitHub Actions环境准备、Node版本警告及上传artifact（无文件）等常规信息，无法判断具体失败原因，可能为日志截断或作业被外部中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715443

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715515

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715551

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715580

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715587

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问的 blob 资源缺失（BlobNotFound），可能是日志上传延迟、文件被清理或路径配置错误，属于基础设施/环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715626

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715673

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971716062

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971716176

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971716185

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971716511

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32221100887/job/95971715668) |


## [Run #32220500767](https://github.com/sgl-project/sglang/actions/runs/32220500767)
- **分支**: `codex/native-encoder-quant-capability`
- **总耗时**: 16.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32220500767

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32220500767/job/95970012028) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node.js弃用警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32220500767/job/95970012028


## [Run #32220091736](https://github.com/sgl-project/sglang/actions/runs/32220091736)
- **分支**: `main`
- **总耗时**: 13.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32220091736

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32220091736/job/95968890378) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32220091736/job/95968890378


## [Run #32216276120](https://github.com/sgl-project/sglang/actions/runs/32216276120)
- **分支**: `main`
- **总耗时**: 60.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32216276120

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 17.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348044) |
| base-b-test-4-npu-a3 / run (0) | 59.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348163) |
| base-b-test-8-npu-a3 / run (0) | 59.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348165) |
| base-b-test-4-npu-a3 / run (1) | 59.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348211) |
| base-b-test-2-npu-a3 / run (0) | 59.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348219) |
| base-b-test-16-npu-a3 / run (0) | 59.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348296) |
| base-b-test-1-npu-a3 / run (0) | 59.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348371) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 59.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348454) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 59.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348455) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 59.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348482) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 59.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348557) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（无文件上传），未包含multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348044

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348163

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348165

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348211

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348219

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348296

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是缓存、依赖或上传步骤异常，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348371

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失，可能是资源被清理或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348454

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348455

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348482

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348557

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32216276120/job/95958348317) |


## [Run #32214123419](https://github.com/sgl-project/sglang/actions/runs/32214123419)
- **分支**: `feat/nixl-deferred-decode-kv-release`
- **总耗时**: 75.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32214123419

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 21.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393241) |
| base-b-test-2-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393247) |
| base-b-test-1-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393260) |
| base-b-test-8-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393267) |
| base-b-test-16-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393277) |
| base-b-test-4-npu-a3 / run (1) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393310) |
| base-b-test-4-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393317) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393527) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393555) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 74.7min | 环境问题 | CI作业因Azure Blob存储中找不到指定blob而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393601) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393669) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败产物或测试未运行到该阶段，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393241

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393247

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393260

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393267

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393277

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查 blob 名称和存储账户状态。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393310

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393317

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393527

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393555

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示BlobNotFound错误，说明作业依赖的某个文件或资源在Azure Blob存储中不存在，可能是上传失败、路径错误或资源被清理，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393601

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393669

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32214123419/job/95952393368) |


## [Run #32213855034](https://github.com/sgl-project/sglang/actions/runs/32213855034)
- **分支**: `main`
- **总耗时**: 30.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32213855034

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 29.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660513) |
| base-b-test-2-npu-a3 / run (0) | 29.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660526) |
| multimodal-gen-test-1-npu-a3 | 9.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660552) |
| base-b-test-8-npu-a3 / run (0) | 29.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660563) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | 环境问题 | CI作业因Azure Blob存储中指定的blob不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660579) |
| base-b-test-1-npu-a3 / run (0) | 29.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660587) |
| base-b-test-4-npu-a3 / run (1) | 29.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660643) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 29.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660766) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 29.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660802) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660824) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 29.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660878) |

- **base-b-test-16-npu-a3 / run (0)**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误（HTTP 404），说明该文件已被删除、路径错误或尚未上传，属于外部依赖资源缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660513

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660526

- **multimodal-gen-test-1-npu-a3**: 日志中未包含multimodal-gen测试的具体执行输出或失败断言，仅显示Node 20弃用警告和上传artifact时无文件。无法确定具体失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660552

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660563

- **base-b-test-4-npu-a3 / run (0)**: 日志显示BlobNotFound错误，表明作业尝试下载或访问的Azure Blob存储资源已被删除或路径错误，属于外部依赖缺失导致的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660579

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储路径及生成步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660587

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660643

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及文件可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660766

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储中找到，可能是资源被清理、路径错误或上传失败，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660802

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660824

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660878

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32213855034/job/95951660581) |


## [Run #32213277921](https://github.com/sgl-project/sglang/actions/runs/32213277921)
- **分支**: `perf/k3_moe-opt`
- **总耗时**: 55.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32213277921

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.0min | 环境问题 | GitHub Actions 下载 actions/checkout 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028204) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028267) |
| base-b-test-1-npu-a3 / run (0) | 54.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028270) |
| base-b-test-4-npu-a3 / run (1) | 54.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028340) |
| base-b-test-8-npu-a3 / run (0) | 54.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028350) |
| base-b-test-4-npu-a3 / run (0) | 54.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028352) |
| base-b-test-16-npu-a3 / run (0) | 54.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028422) |
| base-b-test-2-npu-a3 / run (0) | 54.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028440) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 54.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028593) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 54.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028641) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 54.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028673) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 54.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028695) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/checkout@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功获取该 action，最终作业无法正常执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028204

- **base-a-test-1-npu-a2 / run (0)**: test_mxfp4_situ_output.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，run_suite.py 抛出 ValueError 终止作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028267

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028270

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028340

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象缺失，可能是构建产物未上传、路径错误或存储被清理，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028350

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028352

- **base-b-test-16-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Blob 文件（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028422

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志文件被清理、路径错误或上传失败，属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028440

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028593

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028641

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是资源清理或路径配置错误，需检查相关存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028673

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213277921/job/95950028695


## [Run #32213237737](https://github.com/sgl-project/sglang/actions/runs/32213237737)
- **分支**: `cursor/amd-aiter-diffusion-gqa-d1f1`
- **总耗时**: 37.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32213237737

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 16.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32213237737/job/95949921822) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions运行环境准备、Node版本警告及上传artifact时提示无失败文件。无法判断测试是否真正失败或失败原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32213237737/job/95949921822


## [Run #32212949557](https://github.com/sgl-project/sglang/actions/runs/32212949557)
- **分支**: `perf/k3-mla-q-cache-fusion`
- **总耗时**: 60.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32212949557

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949105841) |
| base-b-test-8-npu-a3 / run (0) | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949105977) |
| base-b-test-2-npu-a3 / run (0) | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106006) |
| base-b-test-4-npu-a3 / run (1) | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106039) |
| base-b-test-16-npu-a3 / run (0) | 60.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106079) |
| base-b-test-1-npu-a3 / run (0) | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106086) |
| base-b-test-4-npu-a3 / run (0) | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106098) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106360) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106364) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106403) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 60.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106493) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或提前退出，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949105841

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成产物，或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949105977

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源（如模型权重或缓存文件）已被删除或路径错误，属于环境配置或资源缺失问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106006

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106039

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106079

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106086

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或依赖文件在 Azure Blob 存储中已被删除或路径错误，属于环境或资源配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106098

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106360

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106364

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106403

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949106493

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212949557/job/95949105991) |


## [Run #32212672686](https://github.com/sgl-project/sglang/actions/runs/32212672686)
- **分支**: `fix-kimi-k3-attn-res-tma-sm120-gate`
- **总耗时**: 94.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32212672686

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 47.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428787) |
| base-b-test-4-npu-a3 / run (1) | 93.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428852) |
| base-b-test-4-npu-a3 / run (0) | 93.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428890) |
| base-b-test-16-npu-a3 / run (0) | 93.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428898) |
| base-b-test-1-npu-a3 / run (0) | 93.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428914) |
| base-b-test-2-npu-a3 / run (0) | 93.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428918) |
| base-b-test-8-npu-a3 / run (0) | 93.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428922) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 93.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960429242) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 93.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960429260) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 93.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960429261) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 93.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960429276) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428787

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428852

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428890

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428898

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428914

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428918

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428922

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）未上传或已被删除，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960429242

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960429260

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960429261

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960429276

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212672686/job/95960428893) |


## [Run #32212644786](https://github.com/sgl-project/sglang/actions/runs/32212644786)
- **分支**: `perf/k3-gfx950-independent-fusions-clean`
- **总耗时**: 35.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32212644786

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248759) |
| base-b-test-16-npu-a3 / run (0) | 34.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248878) |
| base-b-test-8-npu-a3 / run (0) | 34.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248941) |
| base-b-test-4-npu-a3 / run (0) | 34.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248942) |
| base-b-test-4-npu-a3 / run (1) | 34.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248944) |
| base-b-test-2-npu-a3 / run (0) | 34.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248947) |
| base-b-test-1-npu-a3 / run (0) | 34.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248989) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 34.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948249254) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 34.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948249260) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 34.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948249291) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 34.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948249380) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或失败原因被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248759

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248878

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248941

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248942

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248944

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248947

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248989

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948249254

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948249260

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948249291

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948249380

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212644786/job/95948248966) |


## [Run #32212629941](https://github.com/sgl-project/sglang/actions/runs/32212629941)
- **分支**: `kpham/mamba-track-interval-tree-page`
- **总耗时**: 201.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32212629941

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206192) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.4min | 环境问题 | 自定义容器执行失败，runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206701) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 10.5min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95981520873) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206192

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在06:55:06后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206701

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，错误为'Executing the custom container implementation failed'，属于runner环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95981520873

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206371) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206400) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206404) |
| base-b-test-16-npu-a3 / run (0) | 59.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206418) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206421) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206432) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206475) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206784) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 34.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206786) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 71.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95948206813) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 33.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95969032554) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 48.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212629941/job/95970152672) |


## [Run #32212503618](https://github.com/sgl-project/sglang/actions/runs/32212503618)
- **分支**: `refactor/mem-cache-poolhost-dsa`
- **总耗时**: 54.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32212503618

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境警告和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859469) |
| base-b-test-8-npu-a3 / run (0) | 53.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859590) |
| base-b-test-1-npu-a3 / run (0) | 53.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859616) |
| base-b-test-2-npu-a3 / run (0) | 53.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859667) |
| base-b-test-16-npu-a3 / run (0) | 53.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859709) |
| base-b-test-4-npu-a3 / run (0) | 53.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859719) |
| base-b-test-4-npu-a3 / run (1) | 53.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859771) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 53.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859882) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 53.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859889) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 53.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859915) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 53.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947860011) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示Node.js 20弃用警告及upload-artifact未找到文件，未出现明确错误或失败断言，需查看完整日志定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859469

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859590

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859616

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859667

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859709

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859719

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859771

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859882

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859889

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859915

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947860011

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212503618/job/95947859641) |


## [Run #32212209374](https://github.com/sgl-project/sglang/actions/runs/32212209374)
- **分支**: `main`
- **总耗时**: 27.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32212209374

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040000) |
| base-b-test-1-npu-a3 / run (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040041) |
| base-b-test-8-npu-a3 / run (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040057) |
| base-b-test-16-npu-a3 / run (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040098) |
| base-b-test-4-npu-a3 / run (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040100) |
| base-b-test-4-npu-a3 / run (1) | 26.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040123) |
| base-b-test-2-npu-a3 / run (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040135) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040474) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040475) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040542) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040573) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node.js版本警告、上传artifact（未找到文件）及清理步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040000

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存、模型权重或日志文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040041

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040057

- **base-b-test-16-npu-a3 / run (0)**: 作业运行失败，但日志系统尝试从 Azure Blob 下载日志时返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施/存储配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040098

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040100

- **base-b-test-4-npu-a3 / run (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040123

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040135

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040474

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040475

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040542

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040573

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212209374/job/95947040248) |


## [Run #32212182766](https://github.com/sgl-project/sglang/actions/runs/32212182766)
- **分支**: `perf/k3_fused_kda_decode`
- **总耗时**: 219.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32212182766

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.6min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志被截断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963741) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.2min | 环境问题 | GitHub API 返回 500 错误导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95975325662) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 34.9min | 性能回归 | NPU性能测试中qwen3_6_27b用例失败，未达50ms性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95976514214) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，且中间关键测试日志被省略，无法判断具体失败原因，可能为测试未运行或日志记录不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963741

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在调用 GitHub API 查询 lint check 状态时收到 500 服务器错误，属于 GitHub 服务端临时故障，并非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95975325662

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试套件中qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py退出码1，耗时672秒，未通过性能基准，其余用例均通过，判定为性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95976514214

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963848) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963862) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963887) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963895) |
| base-b-test-4-npu-a3 / run (1) | 13.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963950) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963980) |
| base-b-test-16-npu-a3 / run (0) | 83.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946963987) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 53.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946964109) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946964157) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946964190) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95946964225) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95964034173) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32212182766/job/95968388316) |


## [Run #32211161960](https://github.com/sgl-project/sglang/actions/runs/32211161960)
- **分支**: `fix-dcp-args-validation`
- **总耗时**: 45.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32211161960

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076685) |
| base-b-test-1-npu-a3 / run (0) | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076705) |
| base-b-test-8-npu-a3 / run (0) | 44.8min | 环境问题 | CI 作业因 Azure Blob 存储中指定 blob 不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076774) |
| base-b-test-4-npu-a3 / run (1) | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076775) |
| base-b-test-4-npu-a3 / run (0) | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076791) |
| base-b-test-2-npu-a3 / run (0) | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076896) |
| base-b-test-16-npu-a3 / run (0) | 44.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076902) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944077008) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944077075) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944077119) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944077152) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅显示Node 20弃用警告、上传artifact时无文件等非致命信息，实际失败原因被截断或未记录。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076685

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存失效或资源清理导致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076705

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件（如模型权重、缓存或构建产物）在存储中缺失，可能是上传失败、路径错误或过期清理导致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076774

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076775

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076791

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076896

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076902

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944077008

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储中找到，可能是资源被清理或路径配置错误，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944077075

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944077119

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944077152

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32211161960/job/95944076945) |


## [Run #32210758224](https://github.com/sgl-project/sglang/actions/runs/32210758224)
- **分支**: `codex/fix-masked-kv-constexpr`
- **总耗时**: 195.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32210758224

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 其他 | 作业未执行实际测试，仅上传空产物后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908143) |
| base-b-test-16-npu-a3 / run (0) | 1.6min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908232) |
| base-b-test-2-npu-a3 / run (0) | 7.7min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908250) |
| base-b-test-1-npu-a3 / run (0) | 7.6min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908307) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.3min | 其他 | PR健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908378) |
| base-b-test-8-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908402) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908427) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 环境问题 | PR测试健康检查失败，检测到其他根因作业失败导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95965668159) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.2min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95975668148) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未发现测试执行记录，且diffusion-failures目录无文件，最终正常清理退出，无明确失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908143

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到base-b-test-1和base-b-test-2作业失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908232

- **base-b-test-2-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本执行失败，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908250

- **base-b-test-1-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908307

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到base-b-test-2-npu-a3和base-b-test-1-npu-a3两个根因作业失败，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908378

- **base-b-test-8-npu-a3 / run (0)**: 日志显示health-check检测到base-b-test-1和base-b-test-2作业失败，被判定为根因，导致本作业（base-b-test-8）在启动前被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908402

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到base-b-test-2-npu-a3和base-b-test-1-npu-a3两个根因作业失败，触发了fast-fail机制，本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908427

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现base-b-test-2-npu-a3和base-b-test-1-npu-a3两个根因作业失败，本作业作为级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95965668159

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业因健康检查检测到根因作业（base-b-test-2-npu-a3和base-b-test-1-npu-a3）失败而被快速失败，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95975668148

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908229) |
| base-b-test-4-npu-a3 / run (0) | 37.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908295) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908298) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 47.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908458) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 102.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210758224/job/95942908476) |


## [Run #32210674597](https://github.com/sgl-project/sglang/actions/runs/32210674597)
- **分支**: `cpu-shm-process-group`
- **总耗时**: 190.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32210674597

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.8min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志不完整。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352208) |
| base-b-test-16-npu-a3 / run (0) | 9.7min | 环境问题 | rustup 下载超时导致 CI 失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352333) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95960852684) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95965510008) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.2min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95971095275) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.2min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95973775055) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，未包含任何测试执行或失败信息，且上传路径无文件。可能因日志截断或作业提前结束，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352208

- **base-b-test-16-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致安装失败，属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352333

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动前的健康检查中发现根因失败作业base-b-test-16-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95960852684

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动前的健康检查中发现根因作业base-b-test-16-npu-a3失败，触发快速失败机制，本作业被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95965510008

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动前的PR健康检查中，检测到根因作业base-b-test-16-npu-a3失败，触发fast-fail机制，本作业未实际运行即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95971095275

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-b-test-16-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95973775055

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352272) |
| base-b-test-8-npu-a3 / run (0) | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352294) |
| base-b-test-1-npu-a3 / run (0) | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352331) |
| base-b-test-4-npu-a3 / run (1) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352374) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352428) |
| base-b-test-2-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352521) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352685) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 62.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352696) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352755) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210674597/job/95943352761) |


## [Run #32210194241](https://github.com/sgl-project/sglang/actions/runs/32210194241)
- **分支**: `main`
- **总耗时**: 33.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32210194241

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 33.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298719) |
| multimodal-gen-test-1-npu-a3 | 12.4min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298721) |
| base-b-test-4-npu-a3 / run (0) | 33.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298729) |
| base-b-test-4-npu-a3 / run (1) | 33.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298741) |
| base-b-test-1-npu-a3 / run (0) | 33.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298774) |
| base-b-test-16-npu-a3 / run (0) | 33.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298792) |
| base-b-test-2-npu-a3 / run (0) | 33.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298820) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 33.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298906) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 33.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298921) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 33.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298945) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 33.1min | 环境问题 | CI作业因Azure Blob存储中找不到指定blob而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298975) |

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/缓存文件在存储中缺失，可能是由于文件被清理、路径错误或上传失败，属于基础设施环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298719

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、失败断言或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298721

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298729

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298741

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失的环境问题，需检查存储配置或资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298774

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298792

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298820

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298906

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298921

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298945

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示BlobNotFound错误，说明作业依赖的某个文件或工件在Azure Blob存储中不存在，可能是上传失败、路径错误或资源被清理，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298975

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210194241/job/95941298768) |


## [Run #32210047073](https://github.com/sgl-project/sglang/actions/runs/32210047073)
- **分支**: `feature/unified-runtime-attach`
- **总耗时**: 185.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32210047073

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851881) |
| base-b-test-8-npu-a3 / run (0) | 6.4min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851948) |
| base-b-test-16-npu-a3 / run (0) | 2.5min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852096) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852169) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852184) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95962511906) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95970486605) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件（if-no-files-found: ignore），以及Node 20弃用警告。实际失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851881

- **base-b-test-8-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rust-1.92 的校验文件超时，导致 rustup-init 安装失败，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851948

- **base-b-test-16-npu-a3 / run (0)**: 本作业在启动阶段被健康检查快速失败，原因是根因作业base-b-test-8-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852096

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-8-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852169

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查发现根因作业base-b-test-8-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即被终止，属于级联失败而非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852184

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-b-test-8-npu-a3作业失败，被判定为根因，本作业因快速失败机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95962511906

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因作业 base-b-test-8-npu-a3 / run (0) 失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95970486605

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851920) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851938) |
| base-b-test-4-npu-a3 / run (1) | 16.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851940) |
| base-b-test-2-npu-a3 / run (0) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851955) |
| base-b-test-4-npu-a3 / run (0) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940851972) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 91.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852112) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32210047073/job/95940852187) |


## [Run #32209930978](https://github.com/sgl-project/sglang/actions/runs/32209930978)
- **分支**: `perf/k3_fused_kda_decode`
- **总耗时**: 37.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32209930978

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 37.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502185) |
| multimodal-gen-test-1-npu-a3 | 6.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502230) |
| base-b-test-1-npu-a3 / run (0) | 37.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502232) |
| base-b-test-16-npu-a3 / run (0) | 37.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502254) |
| base-b-test-4-npu-a3 / run (1) | 37.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502262) |
| base-a-test-1-npu-a2 / run (0) | 1.0min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502272) |
| base-b-test-4-npu-a3 / run (0) | 37.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502292) |
| base-b-test-8-npu-a3 / run (0) | 37.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502342) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502479) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 37.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502489) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 37.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502506) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 37.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502971) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502185

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅有GitHub Actions环境准备、Node版本警告及上传artifact时未找到失败文件。实际失败原因可能被截断或未记录。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502230

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502232

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/日志文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502254

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502262

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502272

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，请求的资源在存储中未找到。这通常是 CI 配置中引用的工件或缓存文件被删除或路径错误，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502292

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502342

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502479

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的预构建产物或缓存文件未上传或已被删除，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502489

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502506

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209930978/job/95940502971


## [Run #32209504383](https://github.com/sgl-project/sglang/actions/runs/32209504383)
- **分支**: `fix-kimi-k3-attn-res-tma-sm120-gate`
- **总耗时**: 43.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32209504383

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.1min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940982930) |
| base-b-test-4-npu-a3 / run (1) | 42.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983021) |
| base-b-test-2-npu-a3 / run (0) | 42.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983072) |
| base-b-test-4-npu-a3 / run (0) | 42.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983078) |
| base-b-test-8-npu-a3 / run (0) | 42.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983079) |
| base-b-test-16-npu-a3 / run (0) | 42.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983094) |
| base-b-test-1-npu-a3 / run (0) | 42.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983098) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 42.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983259) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983291) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 42.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983297) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 42.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983348) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，虽然后续重试成功，但可能影响作业稳定性。此外，Node 20 弃用警告也提示环境配置需更新。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940982930

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983021

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983072

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983078

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983079

- **base-b-test-16-npu-a3 / run (0)**: 作业在访问Azure Blob存储时返回BlobNotFound错误，说明CI所需的文件或工件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983094

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983098

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983259

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983291

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983297

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983348

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209504383/job/95940983010) |


## [Run #32209218935](https://github.com/sgl-project/sglang/actions/runs/32209218935)
- **分支**: `deepseek_v4_dspark_suppport_pp_pd`
- **总耗时**: 189.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32209218935

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.7min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433325) |
| base-b-test-16-npu-a3 / run (0) | 11.2min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433474) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.9min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433634) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95956856028) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.2min | 其他 | 作业因其他根因作业失败被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95961620096) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95969808684) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行的具体错误。作业最终上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记为失败，需查看完整日志定位真实原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433325

- **base-b-test-16-npu-a3 / run (0)**: 日志显示health-check检测到base-c-test-acc-16-npu-a3作业失败，将其视为根因，导致本作业（base-b-test-16-npu-a3）被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433474

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433634

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到根因作业 base-c-test-acc-16-npu-a3 失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95956856028

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现根因作业 base-c-test-acc-16-npu-a3 失败，触发 fast-fail 机制，本作业被跳过，日志中无自身测试执行记录。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95961620096

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业在启动阶段被健康检查快速失败机制终止，根因是base-c-test-acc-16-npu-a3作业失败，本作业属于级联跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95969808684

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 9.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433446) |
| base-b-test-2-npu-a3 / run (0) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433451) |
| base-b-test-4-npu-a3 / run (1) | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433473) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433488) |
| base-b-test-1-npu-a3 / run (0) | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433500) |
| base-b-test-4-npu-a3 / run (0) | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433571) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 43.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433694) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433695) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32209218935/job/95938433757) |


## [Run #32208880554](https://github.com/sgl-project/sglang/actions/runs/32208880554)
- **分支**: `main`
- **总耗时**: 22.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32208880554

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488032) |
| base-b-test-16-npu-a3 / run (0) | 21.6min | 环境问题 | 日志显示Azure Blob存储中找不到指定文件，属于资源缺失问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488057) |
| base-b-test-4-npu-a3 / run (1) | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488089) |
| base-a-test-1-npu-a2 / run (0) | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488120) |
| base-b-test-1-npu-a3 / run (0) | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488124) |
| base-b-test-4-npu-a3 / run (0) | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488152) |
| base-b-test-8-npu-a3 / run (0) | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488176) |
| base-b-test-2-npu-a3 / run (0) | 21.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488184) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488304) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488371) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488399) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488512) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488032

- **base-b-test-16-npu-a3 / run (0)**: 作业失败原因是下载或访问Azure Blob存储中的blob时返回BlobNotFound错误，说明所需文件不存在或路径错误，可能是CI配置或依赖资源未正确上传，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488057

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488089

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488120

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488124

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488152

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488176

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未上传或路径错误，属于环境配置或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488184

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488304

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488371

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488399

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208880554/job/95937488512


## [Run #32208402261](https://github.com/sgl-project/sglang/actions/runs/32208402261)
- **分支**: `cursor/fix-gfx950-fp8-kv-triton-attention-crash-4ddf`
- **总耗时**: 72.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32208402261

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 72.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696207) |
| base-b-test-16-npu-a3 / run (0) | 72.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696208) |
| base-b-test-1-npu-a3 / run (0) | 72.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696263) |
| base-b-test-4-npu-a3 / run (1) | 72.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696278) |
| base-b-test-2-npu-a3 / run (0) | 72.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696283) |
| base-b-test-4-npu-a3 / run (0) | 72.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696341) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 72.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696533) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 72.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696545) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 72.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696580) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 72.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696593) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696207

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696208

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696263

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696278

- **base-b-test-2-npu-a3 / run (0)**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是CI配置中引用的文件被删除、路径错误或存储账户配置变更，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696283

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696341

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696533

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的预构建产物或缓存文件未上传或已被删除，需检查存储路径或重新生成相关资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696545

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696580

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696593

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32208402261/job/95947696320) |


## [Run #32208261134](https://github.com/sgl-project/sglang/actions/runs/32208261134)
- **分支**: `main`
- **总耗时**: 7.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32208261134

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.4min | 环境问题 | 作业在准备阶段因Node.js 20弃用警告而中断，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650048) |
| base-b-test-8-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650149) |
| base-b-test-4-npu-a3 / run (1) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650189) |
| base-b-test-2-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650201) |
| base-b-test-16-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650230) |
| base-b-test-1-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650240) |
| base-b-test-4-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650243) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650383) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650437) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650451) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650459) |

- **multimodal-gen-test-1-npu-a3**: GitHub Actions runner提示Node.js 20已弃用，强制使用Node.js 24运行actions/checkout和upload-artifact，导致作业在初始化阶段失败，未执行多模态生成测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650048

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650149

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650189

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650201

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650230

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650240

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650243

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650383

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650437

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650451

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650459

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32208261134/job/95935650226) |


## [Run #32208109042](https://github.com/sgl-project/sglang/actions/runs/32208109042)
- **分支**: `codex/minimax-h3-mps`
- **总耗时**: 52.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32208109042

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 47.2min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32208109042/job/95935219306) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含multimodal-gen测试的具体执行输出或错误信息，仅显示actions下载超时重试、Node 20弃用警告及diffusion-failures目录无文件上传。无法判断测试是否失败，可能因日志截断或作业未真正运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32208109042/job/95935219306


## [Run #32207835665](https://github.com/sgl-project/sglang/actions/runs/32207835665)
- **分支**: `model-serve/encoder-internvl-xpu`
- **总耗时**: 136.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32207835665

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460815) |
| base-b-test-4-npu-a3 / run (1) | 1.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460854) |
| multimodal-gen-test-1-npu-a3 | 21.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460861) |
| base-b-test-2-npu-a3 / run (0) | 2.9min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460893) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460899) |
| base-b-test-1-npu-a3 / run (0) | 1.9min | 环境问题 | rustup 下载超时导致 CI 失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460970) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.1min | 其他 | 作业因健康检查快速失败被跳过，非自身测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934461114) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934461155) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934461164) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.4min | 其他 | PR测试健康检查失败，因其他根因作业失败导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934461165) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤掉级联失败后，根因失败作业为base-b-test-2和base-b-test-1，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460815

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业（base-b-test-1/2-npu-a3），本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460854

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js弃用警告和上传artifact时无文件提示，无法判断具体失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460861

- **base-b-test-2-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rust-1.92 的校验文件超时，导致脚本退出，作业失败。属于网络或缓存服务暂时不可用的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460893

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到base-b-test-1/2-npu-a3等根因作业失败，本作业作为级联失败被过滤，最终因快速失败策略终止，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460899

- **base-b-test-1-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致安装失败，属于临时网络或缓存服务问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460970

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示该作业在启动前被PR健康检查机制判定为级联失败而跳过，根因是其他作业（base-b-test-1/2-npu-a3）失败，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934461114

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动前的健康检查中检测到base-b-test-1/2-npu-a3作业失败，被判定为根因失败，因此本作业被fast-fail跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934461155

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-1/2-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止，属于级联跳过而非自身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934461164

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业（base-b-test-1/2-npu-a3），本作业作为级联失败被过滤后触发fast-fail，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934461165

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 33.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460817) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207835665/job/95934460866) |


## [Run #32207742803](https://github.com/sgl-project/sglang/actions/runs/32207742803)
- **分支**: `main`
- **总耗时**: 8.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32207742803

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235518) |
| multimodal-gen-test-1-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235552) |
| base-b-test-8-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235560) |
| base-b-test-16-npu-a3 / run (0) | 8.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235574) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235579) |
| base-b-test-1-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235610) |
| base-b-test-4-npu-a3 / run (1) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235633) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235776) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235790) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235836) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235879) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235518

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235552

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235560

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象已被删除或路径错误，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235574

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235579

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235610

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235633

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235776

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235790

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235836

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235879

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207742803/job/95934235612) |


## [Run #32207601263](https://github.com/sgl-project/sglang/actions/runs/32207601263)
- **分支**: `codex/diffusion-use-checkpoint-quant-spec`
- **总耗时**: 32.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32207601263

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.2min | 环境问题 | GitHub Actions 下载 actions/checkout 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207601263/job/95934165803) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/checkout@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功下载，最终作业无法正常执行。这属于网络或 GitHub 服务问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207601263/job/95934165803


## [Run #32207387118](https://github.com/sgl-project/sglang/actions/runs/32207387118)
- **分支**: `gaopengf/enable_more_topk_for_xpu`
- **总耗时**: 97.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32207387118

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206446) |
| base-b-test-1-npu-a3 / run (0) | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206473) |
| base-b-test-16-npu-a3 / run (0) | 96.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206518) |
| base-b-test-4-npu-a3 / run (0) | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206614) |
| base-b-test-4-npu-a3 / run (1) | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206620) |
| base-b-test-2-npu-a3 / run (0) | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206623) |
| base-b-test-8-npu-a3 / run (0) | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206686) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206787) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206788) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206790) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 96.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206800) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到具体测试命令或错误输出。仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业仍标记为失败，需查看完整日志定位真实原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206446

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206473

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206518

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206614

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206620

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206623

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206686

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206787

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206788

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206790

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206800

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207387118/job/95933206483) |


## [Run #32207382259](https://github.com/sgl-project/sglang/actions/runs/32207382259)
- **分支**: `claude/mm-processor-concurrency-default`
- **总耗时**: 126.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32207382259

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 33.0min | 其他 | 日志不完整，未显示测试失败的具体原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221426) |
| base-b-test-2-npu-a3 / run (0) | 1.2min | 其他 | 该作业因其他根因作业失败而被快速失败（fast-fail）跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221582) |
| base-b-test-4-npu-a3 / run (1) | 1.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221682) |
| base-b-test-16-npu-a3 / run (0) | 2.7min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221697) |
| base-b-test-1-npu-a3 / run (0) | 1.3min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221706) |
| base-b-test-4-npu-a3 / run (0) | 1.4min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221712) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因失败任务，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221893) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.0min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933222097) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.2min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933222110) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 2.3min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933222180) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933222245) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含作业启动、环境准备和上传artifact步骤，未展示实际测试执行过程及失败断言。上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221426

- **base-b-test-2-npu-a3 / run (0)**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤后触发fast-fail机制，提前终止执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221582

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-4-npu-a3 / run (1)）因级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221682

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221697

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被Fast-fail跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221706

- **base-b-test-4-npu-a3 / run (0)**: 作业启动后健康检查发现multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业按快速失败策略跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221712

- **base-b-test-8-npu-a3 / run (0)**: 该作业因健康检查检测到根因失败任务multimodal-gen-test-1-npu-a3而快速失败，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221893

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是多模态生成测试失败，本作业因依赖该测试被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933222097

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933222110

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933222180

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933222245

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207382259/job/95933221655) |


## [Run #32207210932](https://github.com/sgl-project/sglang/actions/runs/32207210932)
- **分支**: `enable_audio_model_transcription`
- **总耗时**: 230.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32207210932

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 38.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707519) |
| base-b-test-16-npu-a3 / run (0) | 4.4min | 环境问题 | rustup下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707664) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.2min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707888) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 5.6min | 环境问题 | GitHub API 请求失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95953073758) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 2.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95958035367) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95970070985) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707519

- **base-b-test-16-npu-a3 / run (0)**: 在安装Rust工具链时，从内部缓存服务下载rustup组件超时，导致脚本执行失败，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707664

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-16-npu-a3/run(0)，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707888

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: github-script 步骤调用 GitHub API 查询 lint check-runs 时返回 500 错误，属于 GitHub 服务端临时故障或限流，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95953073758

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因作业 base-b-test-16-npu-a3 / run (0) 失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95958035367

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-16-npu-a3，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95970070985

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707497) |
| base-b-test-8-npu-a3 / run (0) | 12.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707567) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707580) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707648) |
| base-b-test-4-npu-a3 / run (1) | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707667) |
| base-b-test-1-npu-a3 / run (0) | 27.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707780) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707794) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707800) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32207210932/job/95932707814) |


## [Run #32206889361](https://github.com/sgl-project/sglang/actions/runs/32206889361)
- **分支**: `br_krope_conti`
- **总耗时**: 225.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32206889361

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 20.0min | 环境问题 | GitHub Actions 下载 upload-artifact 超时，导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833426) |
| base-b-test-4-npu-a3 / run (0) | 1.3min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833501) |
| base-b-test-16-npu-a3 / run (0) | 1.8min | 环境问题 | Kubernetes Pod 启动失败，导致作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833504) |
| base-b-test-1-npu-a3 / run (0) | 3.5min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833509) |
| base-b-test-4-npu-a3 / run (1) | 1.0min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833589) |
| base-b-test-2-npu-a3 / run (0) | 2.8min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833590) |
| base-b-test-8-npu-a3 / run (0) | 1.3min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833621) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833996) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 2.6min | 其他 | 作业因其他根因作业失败被快速失败机制跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95951364657) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 3.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95954696697) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 环境问题 | PR测试健康检查失败，根因是其他基础测试作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95966437156) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功下载该 action，最终导致作业无法正常执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833426

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到base-b-test-2-npu-a3作业失败，将其视为根因作业，因此本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833501

- **base-b-test-16-npu-a3 / run (0)**: 自定义 runner 在启动 Pod 时失败，Pod 状态为 Failed，可能是镜像拉取失败、资源不足或节点异常，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833504

- **base-b-test-1-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833509

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业为base-b-test-2-npu-a3 / run (0)，本作业（base-b-test-4-npu-a3）因级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833589

- **base-b-test-2-npu-a3 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业失败。属于网络或缓存服务临时故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833590

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示根因失败作业为base-b-test-2-npu-a3，本作业因级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833621

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动阶段被健康检查拦截，检测到根因作业base-b-test-2-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833996

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因作业base-b-test-16/1/2-npu-a3失败，触发fast-fail跳过本作业，属于级联失败，非本作业自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95951364657

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业在“Check PR test health”步骤因其他根因作业（base-b-test-16/1/2-npu-a3）失败而被快速失败（fast-fail）跳过，并非本作业自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95954696697

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业在启动前的健康检查阶段被快速失败机制跳过，根因是base-b-test-16/1/2-npu-a3等作业失败，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95966437156

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833562) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 97.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833957) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931833962) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206889361/job/95931834029) |


## [Run #32206882148](https://github.com/sgl-project/sglang/actions/runs/32206882148)
- **分支**: `chunyuan/pr_dp_fix`
- **总耗时**: 119.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32206882148

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 38.5min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，可能测试未运行或提前结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931807974) |
| base-b-test-2-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808163) |
| base-b-test-4-npu-a3 / run (0) | 1.3min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联取消 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808255) |
| base-b-test-16-npu-a3 / run (0) | 13.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808268) |
| base-b-test-8-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808360) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 2.6min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808423) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.1min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808447) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.3min | 其他 | 健康检查失败导致级联跳过，根因是base-c-test-acc-2-npu-a3作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808456) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.0min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808531) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业启动后直接进入上传diffusion-failures步骤，但提示无文件可上传，未看到任何测试执行或失败信息，可能测试被跳过或环境初始化失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931807974

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业base-c-test-acc-2-npu-a3失败，因此本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808163

- **base-b-test-4-npu-a3 / run (0)**: 该作业在健康检查阶段检测到根因作业base-c-test-acc-2-npu-a3失败，触发fast-fail机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808255

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是base-c-test-acc-2-npu-a3作业失败，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808268

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤掉级联失败后，根因作业为base-c-test-acc-2-npu-a3，本作业因fast-fail机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808360

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在安装 Rust 工具链时，从内部缓存服务下载 rust-1.92 的 channel 文件超时，导致脚本退出，作业失败。属于网络或缓存服务临时故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808423

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因作业base-c-test-acc-2-npu-a3失败，本作业作为级联失败被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808447

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查将多个作业标记为级联失败，根因作业为base-c-test-acc-2-npu-a3，但当前作业本身未执行测试，属于被级联跳过的间接失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808456

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示本作业未实际运行，而是因PR健康检查检测到根因作业base-c-test-acc-2-npu-a3失败，触发了fast-fail机制，本作业被跳过并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808531

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808172) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808198) |
| base-b-test-1-npu-a3 / run (0) | 28.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206882148/job/95931808229) |


## [Run #32206737801](https://github.com/sgl-project/sglang/actions/runs/32206737801)
- **分支**: `main`
- **总耗时**: 16.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32206737801

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430361) |
| base-b-test-1-npu-a3 / run (0) | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430441) |
| base-b-test-4-npu-a3 / run (0) | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430464) |
| base-b-test-16-npu-a3 / run (0) | 15.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430492) |
| base-b-test-4-npu-a3 / run (1) | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430496) |
| base-b-test-2-npu-a3 / run (0) | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430520) |
| base-b-test-8-npu-a3 / run (0) | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430537) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430671) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430673) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430678) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 15.9min | 环境问题 | CI作业因Azure Blob存储中的日志文件不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430690) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430361

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430441

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430464

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430492

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430496

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430520

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430537

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430671

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430673

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上游产物未上传或已被清理，属于环境或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430678

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业日志显示BlobNotFound错误，说明CI系统尝试下载或访问的日志文件在Azure Blob存储中已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430690

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206737801/job/95931430421) |


## [Run #32206508270](https://github.com/sgl-project/sglang/actions/runs/32206508270)
- **分支**: `glm5.1_enabling`
- **总耗时**: 228.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32206508270

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 31.0min | 其他 | 作业日志被截断，未显示实际失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778768) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95946752593) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 44.1min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，未达性能指标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95951443034) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.3min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95954108223) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95964505020) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时未找到文件，但未给出测试失败或报错信息，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778768

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，耗时1102.71秒，0/1通过，属于性能指标未达到预期要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95946752593

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例退出码1，耗时1317秒，可能因性能未达50ms要求或运行错误导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95951443034

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业在启动前被health-check判定为根因失败（base-c-test-perf-8/16-npu-a3），触发fast-fail跳过，自身未执行测试，属于上游作业失败导致的连带取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95954108223

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业因其他根因作业（base-c-test-perf-8/16-npu-a3）失败而被fast-fail跳过，自身未执行实际测试，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95964505020

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778789) |
| base-b-test-2-npu-a3 / run (0) | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778799) |
| base-a-test-1-npu-a2 / run (0) | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778810) |
| base-b-test-4-npu-a3 / run (0) | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778811) |
| base-b-test-8-npu-a3 / run (0) | 13.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778876) |
| base-b-test-16-npu-a3 / run (0) | 55.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778880) |
| base-b-test-1-npu-a3 / run (0) | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930778931) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930779079) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930779097) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930779124) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 45.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32206508270/job/95930779173) |


## [Run #32205927495](https://github.com/sgl-project/sglang/actions/runs/32205927495)
- **分支**: `cursor/fix-multimodal-gen-1gpu-amd-rocm-d1f1`
- **总耗时**: 137.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32205927495

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 2.5min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154701) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.2min | 其他 | 健康检查快速失败，因同PR中另一作业base-c-test-acc-8-npu-a3失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154919) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 作业被健康检查快速失败，因其他根因作业失败而跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929155015) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.0min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929155064) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.4min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95950362731) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到base-c-test-acc-8-npu-a3作业失败，作为根因作业触发了fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154701

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示health-check检测到根因失败作业base-c-test-acc-8-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止，属于CI依赖链导致的跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154919

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示该作业在启动阶段即被PR健康检查机制判定为级联失败，根因是base-c-test-acc-8-npu-a3作业失败，本作业被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929155015

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致脚本退出码非零，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929155064

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-acc-8-npu-a3作业失败，作为根因作业，导致本作业被快速失败（fast-fail）跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95950362731

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 31.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154684) |
| base-b-test-1-npu-a3 / run (0) | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154692) |
| base-b-test-4-npu-a3 / run (1) | 16.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154728) |
| base-b-test-8-npu-a3 / run (0) | 10.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154747) |
| base-a-test-1-npu-a2 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154757) |
| base-b-test-2-npu-a3 / run (0) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154854) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205927495/job/95929154927) |


## [Run #32205810757](https://github.com/sgl-project/sglang/actions/runs/32205810757)
- **分支**: `main`
- **总耗时**: 15.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32205810757

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973359) |
| base-b-test-4-npu-a3 / run (0) | 14.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973393) |
| base-b-test-16-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973396) |
| base-b-test-2-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973420) |
| multimodal-gen-test-1-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973427) |
| base-b-test-1-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973441) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973449) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973695) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973717) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973727) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973750) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973359

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是上传失败、路径错误或资源被删除，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973393

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973396

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973420

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973427

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973441

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973449

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973695

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973717

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973727

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973750

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205810757/job/95928973439) |


## [Run #32205599687](https://github.com/sgl-project/sglang/actions/runs/32205599687)
- **分支**: `lsyin/page-tail-write`
- **总耗时**: 202.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32205599687

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269030) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 35.7min | 性能回归 | NPU性能测试中deepseek_v4_flash用例未达性能目标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95949848790) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 8.7min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95958295527) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅显示Node.js版本弃用警告和上传artifact时无文件。可能因日志截断或作业在测试前被取消，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269030

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms用例退出码1，运行380秒后失败，疑似性能未达50ms延迟目标，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95949848790

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-16-npu-a3作业失败，被判定为根因作业，导致本作业（base-c-test-perf-2-npu-a3）被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95958295527

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 58.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269015) |
| base-b-test-8-npu-a3 / run (0) | 9.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269029) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269078) |
| base-b-test-4-npu-a3 / run (0) | 34.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269081) |
| base-b-test-2-npu-a3 / run (0) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269114) |
| base-b-test-4-npu-a3 / run (1) | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269207) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269295) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269299) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 45.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269374) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269377) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95928269379) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95944292032) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 26.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32205599687/job/95950304718) |


## [Run #32205486969](https://github.com/sgl-project/sglang/actions/runs/32205486969)
- **分支**: `fmha_v2`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32205486969

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 5.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886599) |
| base-b-test-1-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886602) |
| base-b-test-16-npu-a3 / run (0) | 5.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886611) |
| multimodal-gen-test-1-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886615) |
| base-b-test-4-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886630) |
| base-b-test-4-npu-a3 / run (1) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886650) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | 环境问题 | 自定义容器执行失败，测试中途被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886736) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886773) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886778) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886838) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886857) |
| base-b-test-8-npu-a3 / run (0) | 5.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927887154) |

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是日志上传或依赖下载路径错误，属于基础设施或配置问题，与代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886599

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理或路径错误，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886602

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或文件被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886611

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886615

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886630

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886650

- **base-a-test-1-npu-a2 / run (0)**: 第二个测试文件test_npu_ascend_dsv4_backend.py刚开始执行时，自定义容器实现报错（Executing the custom container implementation failed），导致作业提前终止，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886736

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886773

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886778

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886838

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927886857

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储资源缺失，可能是构建产物或依赖文件未正确上传，或路径配置错误。建议检查CI流程中相关存储路径及上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/32205486969/job/95927887154


---
*Auto-generated by npu_pr_monitor.py*