# NPU CI 执行监控
**生成时间**: 2026-08-17 00:37 UTC
**分析 Run 数**: 36

---

## 📊 本次执行总结

- **成功 Job 数**: 246
- **失败 Run 数**: 36
- **成功 Job 平均耗时**: 27.8min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 275.6min | #31958834769 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95197311042) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 269.4min | #31955324456 | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95188400280) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 263.5min | #31941258191 | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190430372) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 132.4min | #31941258191 | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429528) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 110.7min | #31956934352 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919359) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.6min | #31950203437 | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172448086) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.3min | #31970230694 | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221498928) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.2min | #31958260416 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213551) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.4min | #31955324456 | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036397) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.0min | #31951691163 | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043816) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31958834769<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `codex/diffusion-reuse-srt-qwen-vision` | 308.7min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31958834769) |
| #31955324456<br>[#34565 [Unified Tree] Support Branching-Point Caching for the SWA Component](https://github.com/sgl-project/sglang/pull/34565) | `cjc/unified-swa-branching` | 298.1min | 4 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31955324456) |
| #31950203437<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `codex/diffusion-reuse-srt-qwen-vision` | 176.9min | 4 | multimodal-gen-test-1-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31950203437) |
| #31958260416<br>[#35004 [Diffusion] Reuse SRT CLIP encoder blocks](https://github.com/sgl-project/sglang/pull/35004) | `codex/diffusion-reuse-srt-clip` | 115.9min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31958260416) |
| #31970230694<br>[#33778 Avoid materializing GDN QKV tensors during target verification](https://github.com/sgl-project/sglang/pull/33778) | `perf/gdn-strided-target-verify` | 113.7min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31970230694) |
| #31956934352<br>[#33561 [Model] Support Ling-3.0-flash (BailingMoeV3) ](https://github.com/sgl-project/sglang/pull/33561) | `ling3-flash-dspark` | 113.3min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31956934352) |
| #31951691163<br>[#35031 [JIT Kernel] Migrate causal_conv1d_fwd and causal_conv1d_update from AOT to JIT](https://github.com/sgl-project/sglang/pull/35031) | `mmangkad/migrate-causal-conv1d-jit` | 93.1min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31951691163) |
| #31942715216<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `codex/diffusion-reuse-srt-qwen-vision` | 92.6min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31942715216) |
| #31952159941<br>[#35034 [VLM] Add preprocess-cache observability and agentic benchmark coverage](https://github.com/sgl-project/sglang/pull/35034) | `codex/k3-mm-cache-lease` | 90.8min | 4 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31952159941) |
| #31955054550<br>[#35020 [Fix] Correct dense FP8 Marlin bias ordering](https://github.com/sgl-project/sglang/pull/35020) | `fix-qwen2-marlin-fp8-bias` | 89.5min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31955054550) |
| #31947333589<br>[#35034 [VLM] Add preprocess-cache observability and agentic benchmark coverage](https://github.com/sgl-project/sglang/pull/35034) | `codex/k3-mm-cache-lease` | 88.9min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31947333589) |
| #31959114825<br>[#35042 Fix sconv track refresh on graph capture](https://github.com/sgl-project/sglang/pull/35042) | `fix-sconv-track-refresh-on-graph-capture` | 87.7min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31959114825) |
| #31953585741<br>[#35004 [Diffusion] Reuse SRT CLIP encoder blocks](https://github.com/sgl-project/sglang/pull/35004) | `codex/diffusion-reuse-srt-clip` | 87.3min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31953585741) |
| #31946070244<br>[#34988 [Diffusion] Reuse SRT SigLIP vision model](https://github.com/sgl-project/sglang/pull/34988) | `codex/diffusion-reuse-srt-siglip` | 86.4min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31946070244) |
| #31955085141<br>[#34967 [MoE] Add FlashInfer SM90 MXFP4 W4A8 CUTLASS MoE](https://github.com/sgl-project/sglang/pull/34967) | `flashinfer-sm90-mxfp4-fp8` | 85.7min | 3 | multimodal-gen-test-1-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31955085141) |
| #31947880093<br>[#35004 [Diffusion] Reuse SRT CLIP encoder blocks](https://github.com/sgl-project/sglang/pull/35004) | `codex/diffusion-reuse-srt-clip` | 85.6min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31947880093) |
| #31941258191<br>[#30319 [NPU] Add mxfp4-w4a4 MOE Quantization Support for NPU](https://github.com/sgl-project/sglang/pull/30319) | `add_mxfp4w4a4_quantization_for_npu` | 81.5min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31941258191) |
| #31965789132<br>[#35050 [XPU] Fix decode graph runner is_current_stream_capturing on non-CUDA devices](https://github.com/sgl-project/sglang/pull/35050) | `fix/xpu-decode-graph-runner-is-current-stream-capturing` | 69.4min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31965789132) |
| #31948344563<br>[#35016 [diffusion] test: tighten NVIDIA perf baselines](https://github.com/sgl-project/sglang/pull/35016) | `main` | 44.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31948344563) |
| #31945849046<br>[#35016 [diffusion] test: tighten NVIDIA perf baselines](https://github.com/sgl-project/sglang/pull/35016) | `codex/tighten-nv-perf-baselines-20260816` | 41.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31945849046) |
| #31946576010<br>[#34932 [diffusion] Accelerate Cosmos3 T2I QKNorm+RoPE](https://github.com/sgl-project/sglang/pull/34932) | `main` | 39.6min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31946576010) |
| #31946164889<br>[#35004 [Diffusion] Reuse SRT CLIP encoder blocks](https://github.com/sgl-project/sglang/pull/35004) | `codex/diffusion-reuse-srt-clip` | 38.3min | 6 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31946164889) |
| #31976116354<br>[#35058 [Spec] Simplify compute_spec_v2_logprobs signature and skip identity gathers](https://github.com/sgl-project/sglang/pull/35058) | `lsyin/spec-logprob-args` | 33.1min | 7 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-b-test-16-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31976116354) |
| #31977956528<br>[#35058 [Spec] Simplify compute_spec_v2_logprobs signature and skip identity gathers](https://github.com/sgl-project/sglang/pull/35058) | `main` | 29.9min | 9 | base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31977956528) |
| #31976583057<br>[#34996 Increase post-capture decode memory reserve](https://github.com/sgl-project/sglang/pull/34996) | `main` | 26.0min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31976583057) |
| #31945459138<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `main` | 15.2min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31945459138) |
| #31956957774<br>[#32779 [SM120&90] Add CUDA fused Triton sparse-MLA prefill backend for DSA](https://github.com/sgl-project/sglang/pull/32779) | `dsa-triton-sparse-mla-prefill` | 13.6min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31956957774) |
| #31975866108<br>[#34995 [VLM] Avoid synchronizing multimodal placeholder counts](https://github.com/sgl-project/sglang/pull/34995) | `main` | 12.8min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31975866108) |
| #31946133977<br>[#34928 [diffusion][kernel] Accelerate Sana BCG with bit-exact conv post-processing](https://github.com/sgl-project/sglang/pull/34928) | `main` | 10.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31946133977) |
| #31946659232<br>[#35034 [VLM] Add preprocess-cache observability and agentic benchmark coverage](https://github.com/sgl-project/sglang/pull/35034) | `codex/k3-mm-cache-lease` | 8.8min | 11 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-b-test-4-npu-a3 / run (1), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31946659232) |
| #31978978711<br>[#31324 [AMD] [GLM5] Skip DSA decode indexer when kv_len <= index_topk (dense k-only fast path)](https://github.com/sgl-project/sglang/pull/31324) | `jacob/dsa-decode-skip-indexer` | 8.3min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31978978711) |
| #31975021301<br>[#34994 Build Rust extensions on demand in source checkouts](https://github.com/sgl-project/sglang/pull/34994) | `main` | 7.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31975021301) |
| #31973997491<br>[#34982 [misc] Rename shared-read boundary to shared-read ends and fix wrapper delegation](https://github.com/sgl-project/sglang/pull/34982) | `main` | 7.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31973997491) |
| #31947030532<br>[#35034 [VLM] Add preprocess-cache observability and agentic benchmark coverage](https://github.com/sgl-project/sglang/pull/35034) | `codex/k3-mm-cache-lease` | 7.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31947030532) |
| #31975533814<br>[#35001 [Frontend] Apply request header overrides to chat completions](https://github.com/sgl-project/sglang/pull/35001) | `main` | 6.5min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31975533814) |
| #31946415467<br>[#35034 [VLM] Add preprocess-cache observability and agentic benchmark coverage](https://github.com/sgl-project/sglang/pull/35034) | `codex/k3-mm-cache-lease` | 5.9min | 12 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31946415467) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 33 | 30 | 0 | 3 |
| base-b-test-1-npu-a3 / run (0) | 33 | 20 | 0 | 13 |
| base-b-test-16-npu-a3 / run (0) | 33 | 18 | 0 | 15 |
| base-b-test-2-npu-a3 / run (0) | 33 | 20 | 0 | 13 |
| base-b-test-4-npu-a3 / run (0) | 33 | 19 | 0 | 14 |
| base-b-test-4-npu-a3 / run (1) | 33 | 21 | 0 | 12 |
| base-b-test-8-npu-a3 / run (0) | 33 | 25 | 0 | 8 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 33 | 20 | 0 | 13 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 33 | 15 | 0 | 18 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 33 | 18 | 0 | 15 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 33 | 29 | 0 | 4 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 20 | 3 | 0 | 17 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 15 | 1 | 0 | 14 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 18 | 4 | 0 | 14 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 29 | 3 | 0 | 26 |
| multimodal-gen-test-1-npu-a3 | 36 | 0 | 20 | 16 |

---


## [Run #31978978711](https://github.com/sgl-project/sglang/actions/runs/31978978711)
- **分支**: `jacob/dsa-decode-skip-indexer`
- **总耗时**: 8.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31978978711

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711075) |
| base-b-test-4-npu-a3 / run (0) | 7.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711077) |
| base-b-test-2-npu-a3 / run (0) | 7.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711082) |
| base-b-test-1-npu-a3 / run (0) | 7.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711100) |
| base-a-test-1-npu-a2 / run (0) | 2.0min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711116) |
| base-b-test-16-npu-a3 / run (0) | 7.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711138) |
| base-b-test-4-npu-a3 / run (1) | 6.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711191) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711202) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711301) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.1min | 环境问题 | 自定义容器执行失败，导致作业在安装依赖阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711302) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.5min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711308) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711412) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711075

- **base-b-test-4-npu-a3 / run (0)**: 测试在运行第二个测试文件时，自定义容器实现执行失败，提示请联系自托管runner管理员，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711077

- **base-b-test-2-npu-a3 / run (0)**: 日志显示服务启动后健康检查返回503，随后自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711082

- **base-b-test-1-npu-a3 / run (0)**: 作业在运行torchair配置时出现警告，随后自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711100

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rust-1.92 的元数据文件超时，导致脚本退出码非零，作业失败。属于基础设施网络或缓存服务问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711116

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711138

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试用例已通过（OK），但在运行下一个测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711191

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711202

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711301

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在pip安装evalscope依赖时，执行自定义容器实现失败（Executing the custom container implementation failed），可能是容器环境或资源问题，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711302

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在加载模型分片时（282个分片仅加载1个）容器实现报错，导致执行终止。可能是容器镜像或运行环境问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711308

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业base-a-test-1-npu-a2/run(0)，触发fast-fail机制，本作业未实际运行即被跳过，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31978978711/job/95242711412


## [Run #31977956528](https://github.com/sgl-project/sglang/actions/runs/31977956528)
- **分支**: `main`
- **总耗时**: 29.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31977956528

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 28.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230174) |
| base-b-test-1-npu-a3 / run (0) | 28.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230189) |
| multimodal-gen-test-1-npu-a3 | 28.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230264) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 代码错误 | HiCache MLA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230271) |
| base-b-test-2-npu-a3 / run (0) | 28.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230304) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 28.1min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230397) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 28.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230463) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230488) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230495) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载至28%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230174

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行到67%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230189

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅有GitHub Actions运行环境准备、Node版本警告及上传artifact时未找到文件的提示。无法判断测试失败原因，可能为日志截断或作业被外部中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230264

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test_npu_hicache_mla.py在NPU A3环境下运行281秒后失败，退出码为1，导致整个作业终止。具体失败原因需查看该测试文件的详细输出，可能涉及功能实现或环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230271

- **base-b-test-2-npu-a3 / run (0)**: 测试运行到96%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230304

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（decode吞吐约420 token/s），但在23:30:58时容器执行失败，错误为"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230397

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在处理请求时出现“Executing the custom container implementation failed”错误，提示联系runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230463

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-4-npu-a3/run(0)，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230488

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业base-b-test-4-npu-a3失败，本作业作为级联失败被过滤，最终因快速失败策略被跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230495

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230188) |
| base-b-test-8-npu-a3 / run (0) | 10.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230199) |
| base-b-test-4-npu-a3 / run (1) | 26.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31977956528/job/95240230218) |


## [Run #31976583057](https://github.com/sgl-project/sglang/actions/runs/31976583057)
- **分支**: `main`
- **总耗时**: 26.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31976583057

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.0min | 其他 | 日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938533) |
| base-b-test-1-npu-a3 / run (0) | 24.4min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938549) |
| base-b-test-2-npu-a3 / run (0) | 23.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938570) |
| base-b-test-16-npu-a3 / run (0) | 21.6min | 环境问题 | NPU容器执行失败，模型权重加载时发生崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938581) |
| base-b-test-4-npu-a3 / run (1) | 22.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938587) |
| base-b-test-4-npu-a3 / run (0) | 8.3min | 代码错误 | NPU HiCache MLA 测试失败，测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938653) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 22.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938664) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938706) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 21.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938719) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95237771598) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifacts（无文件）等常规信息，未出现测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938533

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载完成后，出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938549

- **base-b-test-2-npu-a3 / run (0)**: 测试运行到91%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境问题而非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938570

- **base-b-test-16-npu-a3 / run (0)**: 在加载MoE模型权重时，torch的copy_操作在NPU上执行失败，导致Scheduler watchdog超时，最终容器执行失败。可能是NPU环境或驱动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938581

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中容器实现执行失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938587

- **base-b-test-4-npu-a3 / run (0)**: 测试 test_npu_hicache_mla.py 执行失败（退出码1），耗时291秒，0/5测试通过。可能是代码逻辑错误或环境配置问题导致测试断言失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938653

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938664

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938706

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但执行自定义容器时失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938719

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在性能测试运行过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95237771598

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938574) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938594) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976583057/job/95236938740) |


## [Run #31976116354](https://github.com/sgl-project/sglang/actions/runs/31976116354)
- **分支**: `lsyin/spec-logprob-args`
- **总耗时**: 33.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31976116354

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 31.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603252) |
| base-b-test-4-npu-a3 / run (0) | 18.6min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603306) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603422) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 16.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603436) |
| base-b-test-16-npu-a3 / run (0) | 23.6min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603505) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95237329820) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 5.2min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95239568082) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件），未包含multimodal-gen测试的具体执行输出或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603252

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试运行约18分钟后，在Prefill/Decode正常处理时突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603306

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603422

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603436

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型加载过程中（加载161个分片时）出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603505

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1118秒，未通过性能基准，可能因模型推理速度或延迟不满足50ms要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95237329820

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示在启动阶段出现“Executing the custom container implementation failed”错误，属于自托管runner环境问题，并非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95239568082

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603265) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603269) |
| base-b-test-1-npu-a3 / run (0) | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603287) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603322) |
| base-b-test-8-npu-a3 / run (0) | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603389) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603406) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31976116354/job/95236603550) |


## [Run #31975866108](https://github.com/sgl-project/sglang/actions/runs/31975866108)
- **分支**: `main`
- **总耗时**: 12.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31975866108

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.8min | 其他 | 日志不完整，未显示测试执行过程，仅看到上传artifact时无失败文件，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192134) |
| base-b-test-2-npu-a3 / run (0) | 11.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192187) |
| base-b-test-1-npu-a3 / run (0) | 9.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192211) |
| base-b-test-4-npu-a3 / run (0) | 8.5min | 环境问题 | 自定义容器执行失败，CUDA coredump 未启用导致进程异常退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192220) |
| base-b-test-16-npu-a3 / run (0) | 7.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192229) |
| base-b-test-4-npu-a3 / run (1) | 8.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192230) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192426) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192438) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.8min | 环境问题 | 自定义容器执行失败，NPU CI 环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192470) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8.3min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235582188) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions基础设施信息（Node版本警告、artifact上传等），未包含实际测试命令输出或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192134

- **base-b-test-2-npu-a3 / run (0)**: 测试逻辑正常执行完毕（HTTP 200），但随后自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192187

- **base-b-test-1-npu-a3 / run (0)**: 作业在运行NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员。日志显示torch_npu相关警告，但核心错误是容器执行问题，属于环境配置或基础设施故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192211

- **base-b-test-4-npu-a3 / run (0)**: 作业在运行测试时触发 CUDA 用户触发的 coredump，但未设置 CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1，导致多个进程等待 coredump 超时，最终容器执行失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192220

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试运行到98%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192229

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行正常（进度69%），但随后出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192230

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在22:28:16出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192426

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192438

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示模型权重加载到36%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU CI环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192470

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在容器内执行环境变量获取后，报错“Executing the custom container implementation failed”，属于自托管runner容器环境异常，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235582188

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192213) |
| base-b-test-8-npu-a3 / run (0) | 11.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192264) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31975866108/job/95235192466) |


## [Run #31975533814](https://github.com/sgl-project/sglang/actions/runs/31975533814)
- **分支**: `main`
- **总耗时**: 6.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31975533814

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.3min | 环境问题 | 作业因缺少diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385473) |
| base-b-test-1-npu-a3 / run (0) | 2.9min | 环境问题 | 自定义容器执行失败，导致作业在安装依赖阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385496) |
| base-b-test-8-npu-a3 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385516) |
| base-b-test-2-npu-a3 / run (0) | 5.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385528) |
| base-b-test-16-npu-a3 / run (0) | 4.9min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385531) |
| base-b-test-4-npu-a3 / run (0) | 3.9min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385556) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385580) |
| base-b-test-4-npu-a3 / run (1) | 2.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385594) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385665) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.3min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385666) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385704) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234900956) |

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤未找到diffusion-failures/目录，说明测试未生成失败样本，作业可能因环境或前置步骤异常而终止，未进入核心测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385473

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在构建sgl-eval包时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，而非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385496

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动NPU容器后，TokenizerManager初始化过程中自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385516

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重时（约50%进度）自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器运行问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385528

- **base-b-test-16-npu-a3 / run (0)**: 日志显示Watchdog TokenizerManager等组件初始化后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境配置或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385531

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动测试容器时失败，错误信息为"Executing the custom container implementation failed"，属于自托管runner环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385556

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行NPU测试时，自定义容器实现执行失败，导致测试进程中断。日志显示测试刚开始执行就报错，属于自托管runner环境问题，而非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385580

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动测试前，执行自定义容器实现时失败，提示联系runner管理员，属于基础设施环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385594

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在加载模型权重时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385665

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在加载模型分片（约12%）时，GitHub Actions 报错“Executing the custom container implementation failed”，属于自托管 runner 容器环境异常，非代码或测试逻辑问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385666

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型权重加载到94%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385704

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示执行自定义容器实现时出错（Executing the custom container implementation failed），提示联系自托管 runner 管理员，属于 runner 或容器环境配置问题，而非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234900956

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31975533814/job/95234385662) |


## [Run #31975021301](https://github.com/sgl-project/sglang/actions/runs/31975021301)
- **分支**: `main`
- **总耗时**: 7.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31975021301

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099469) |
| base-b-test-1-npu-a3 / run (0) | 6.6min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099485) |
| base-b-test-4-npu-a3 / run (0) | 6.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099497) |
| base-b-test-4-npu-a3 / run (1) | 5.3min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099509) |
| base-b-test-8-npu-a3 / run (0) | 6.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099538) |
| base-b-test-16-npu-a3 / run (0) | 4.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099542) |
| base-b-test-2-npu-a3 / run (0) | 6.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099614) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099711) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.7min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099737) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.5min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099740) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.9min | 环境问题 | 自定义容器启动失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233684663) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤。可能因日志截断或作业在测试前被取消，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099469

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载成功，但在执行自定义容器时出现错误，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099485

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在批量捕获测试数据时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU资源或容器环境问题导致测试中断，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099497

- **base-b-test-4-npu-a3 / run (1)**: 日志显示模型权重加载到43%时，runner报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099509

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试运行中突然出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099538

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型权重加载过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099542

- **base-b-test-2-npu-a3 / run (0)**: 作业在初始化torch分布式时失败，错误为"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099614

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载模型权重时（约23%进度）自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099711

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动自定义容器时失败，错误信息为"Executing the custom container implementation failed"，可能是NPU驱动或容器配置问题，导致无法正常运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099737

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示服务启动后，在生成请求时出现NPU算子回退警告，随后自定义容器执行失败，提示联系自托管runner管理员，属于NPU环境配置或容器兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099740

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在安装Rust工具链后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233684663

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099577) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31975021301/job/95233099696) |


## [Run #31973997491](https://github.com/sgl-project/sglang/actions/runs/31973997491)
- **分支**: `main`
- **总耗时**: 7.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31973997491

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647033) |
| base-b-test-1-npu-a3 / run (0) | 7.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647060) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647067) |
| base-b-test-16-npu-a3 / run (0) | 7.0min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647088) |
| base-b-test-4-npu-a3 / run (0) | 6.9min | 环境问题 | 自定义容器执行失败，NPU测试中途退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647105) |
| base-b-test-2-npu-a3 / run (0) | 6.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647110) |
| base-b-test-4-npu-a3 / run (1) | 5.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647157) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.0min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647188) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.3min | 环境问题 | 自定义容器启动失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647213) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.0min | 环境问题 | 自定义容器执行失败，导致作业在启动阶段即中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647289) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 2.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95231176413) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未出现测试执行或失败断言信息，无法判断具体失败原因，可能为日志截断或作业被外部中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647033

- **base-b-test-1-npu-a3 / run (0)**: 日志显示服务启动正常，但随后出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647060

- **base-b-test-8-npu-a3 / run (0)**: 作业在运行约7分钟后，日志显示"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647067

- **base-b-test-16-npu-a3 / run (0)**: 日志显示容器执行失败（Executing the custom container implementation failed），随后进入清理流程，未运行任何测试，属于自托管runner环境或镜像问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647088

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试在捕获批次时正常进行，但随后报错“Executing the custom container implementation failed”，属于自托管runner容器环境异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647105

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在加载模型权重时出现ImportError（无法从mm_utils导入MultimodalDataItem），随后自定义容器实现执行失败，作业被终止。这属于环境或依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647110

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在加载模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647157

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在安装依赖过程中，自定义容器实现执行失败，提示联系自托管 runner 管理员，可能是容器环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647188

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示容器初始化过程中出现导入错误（如MultimodalDataItem导入失败），随后报错“Executing the custom container implementation failed”，作业在测试开始前即被终止，属于运行环境或依赖配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647213

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在运行自定义容器实现时出现错误（Executing the custom container implementation failed），作业未能进入实际测试阶段，属于自托管runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647289

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95231176413

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647236) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31973997491/job/95230647248) |


## [Run #31970230694](https://github.com/sgl-project/sglang/actions/runs/31970230694)
- **分支**: `perf/gdn-strided-target-verify`
- **总耗时**: 113.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31970230694

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221501313) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | NPU性能测试未达预期，测试用例执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95222185744) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 31.3min | 超时 | NPU性能测试超时失败，测试用例执行时间超过预期。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95224285340) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95226611899) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95234684076) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体错误信息，仅显示上传diffusion-failures目录时提示无文件，可能测试未产生失败产物或日志被截断，需查看完整日志定位失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221501313

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1133秒，未通过性能测试，可能因性能未达标或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95222185744

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py运行1463秒后失败，估计时间3600秒，但实际未完成，0/4测试通过，可能因性能未达标或环境问题导致超时。
  链接: https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95224285340

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95226611899

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3、base-c-test-perf-8/16-npu-a3等根因作业失败，本作业被Fast-fail机制跳过，属于级联失败，非本作业自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95234684076

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221498821) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221498912) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221498928) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221498938) |
| base-b-test-16-npu-a3 / run (0) | 52.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221498965) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221499336) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221500135) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221500767) |
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221500778) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221508838) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31970230694/job/95221508906) |


## [Run #31965789132](https://github.com/sgl-project/sglang/actions/runs/31965789132)
- **分支**: `fix/xpu-decode-graph-runner-is-current-stream-capturing`
- **总耗时**: 69.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31965789132

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 35.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传diffusion-failures产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210690864) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 67.8min | 精度回归 | NPU精度测试用例qwen3_5_9b_bf16_1p_gsm8k失败，0/3通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210691026) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.7min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95211845440) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 37.8min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，未达性能标准。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95214322595) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.0min | 其他 | 作业因其他根因作业失败被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95216264711) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，可能测试未产生失败样本或测试提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210690864

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，耗时3848秒，超过预估3600秒，所有3个测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210691026

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1091秒后失败，0/1通过。该测试为性能测试，失败原因可能是性能未达到预设阈值（如50ms延迟目标），需检查具体性能指标是否回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95211845440

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，qwen3_235b_w8a8_8p_in3k5_out1k5_50ms用例退出码1，耗时1443秒，可能因性能未达标或运行错误导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95214322595

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3失败，触发fast-fail机制，本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95216264711

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210690868) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210690879) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210690900) |
| base-b-test-4-npu-a3 / run (0) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210690908) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210690921) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210690952) |
| base-b-test-16-npu-a3 / run (0) | 50.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210691029) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210691075) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210691076) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31965789132/job/95210691141) |


## [Run #31959114825](https://github.com/sgl-project/sglang/actions/runs/31959114825)
- **分支**: `fix-sconv-track-refresh-on-graph-capture`
- **总耗时**: 87.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31959114825

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305795) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95199376724) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 28.7min | 性能回归 | NPU性能测试未通过，0/4用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95201495931) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.1min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95203505198) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95208886820) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node版本弃用警告和上传artifact时无文件提示，无法定位具体失败点，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305795

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1118秒后失败，0/1通过，属于性能测试未达标，可能因模型推理速度或吞吐量低于阈值导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95199376724

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中4个性能测试全部失败，首个失败用例为qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms测试，运行1437秒后退出码1，可能因性能未达预期或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95201495931

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业未实际运行，因同次运行中其他作业（multimodal-gen-test-1-npu-a3 和 base-c-test-perf-8-npu-a3）失败，触发了 fast-fail 跳过机制，导致本作业被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95203505198

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3、base-c-test-perf-8/16-npu-a3等根因作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95208886820

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 48.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305855) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305893) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305911) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305922) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305951) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305990) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198305993) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198306084) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198306098) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198306108) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31959114825/job/95198306156) |


## [Run #31958834769](https://github.com/sgl-project/sglang/actions/runs/31958834769)
- **分支**: `codex/diffusion-reuse-srt-qwen-vision`
- **总耗时**: 308.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31958834769

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193664933) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 70.6min | 精度回归 | NPU精度测试用例qwen3_5_9b_bf16_1p_gsm8k执行失败，0/3测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193665066) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的输出，只有GitHub Actions的初始化、上传artifact（无文件）和清理步骤。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193664933

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，耗时4015秒超过预估3600秒，所有3个精度测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193665066

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193664959) |
| base-b-test-1-npu-a3 / run (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193664968) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193664976) |
| base-a-test-1-npu-a2 / run (0) | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193664977) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193664993) |
| base-b-test-16-npu-a3 / run (0) | 45.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193664998) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193665005) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193665054) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193665055) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95193665063) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95195211244) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 275.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95197311042) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958834769/job/95198093327) |


## [Run #31958260416](https://github.com/sgl-project/sglang/actions/runs/31958260416)
- **分支**: `codex/diffusion-reuse-srt-clip`
- **总耗时**: 115.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31958260416

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213309) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95193321899) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95197215515) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 上游作业失败导致级联跳过，本作业未实际运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95197754901) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95206126388) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213309

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1117秒后返回退出码1，0/1测试通过，属于性能指标未达到预期要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95193321899

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业在启动前，健康检查发现同次运行中另一个作业（base-c-test-perf-8-npu-a3）失败，触发了fast-fail机制，导致本作业被跳过并报错退出，并非本作业自身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95197215515

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业被快速失败机制跳过，日志中无实际测试执行内容。
  链接: https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95197754901

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 作业在启动阶段因PR健康检查发现其他根因作业（multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3）失败，触发fast-fail机制，本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95206126388

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213362) |
| base-b-test-4-npu-a3 / run (1) | 11.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213404) |
| base-b-test-4-npu-a3 / run (0) | 27.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213415) |
| base-b-test-1-npu-a3 / run (0) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213418) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213454) |
| base-b-test-16-npu-a3 / run (0) | 52.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213462) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213523) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213551) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213628) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213690) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31958260416/job/95192213695) |


## [Run #31956957774](https://github.com/sgl-project/sglang/actions/runs/31956957774)
- **分支**: `dsa-triton-sparse-mla-prefill`
- **总耗时**: 13.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31956957774

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987491) |
| base-b-test-16-npu-a3 / run (0) | 12.5min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987556) |
| base-b-test-4-npu-a3 / run (1) | 12.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987561) |
| base-b-test-1-npu-a3 / run (0) | 10.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987577) |
| base-b-test-2-npu-a3 / run (0) | 11.2min | 环境问题 | 自定义容器执行失败，NPU作业在加载权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987584) |
| base-b-test-4-npu-a3 / run (0) | 11.8min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987706) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987719) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987755) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987806) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 5.0min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95189888948) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的输出，只有runner初始化、Node版本警告和artifact上传（无文件）等常规信息。无法判断具体失败原因，可能是日志截断或作业在测试前已异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987491

- **base-b-test-16-npu-a3 / run (0)**: 作业在模型权重加载阶段（DP/TP多进程加载）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987556

- **base-b-test-4-npu-a3 / run (1)**: 作业在加载模型权重时（约16:06:03）自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987561

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在编译token过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987577

- **base-b-test-2-npu-a3 / run (0)**: 作业在模型加载阶段（Load weight begin）后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987584

- **base-b-test-4-npu-a3 / run (0)**: 日志显示容器启动后加载模型时出现多个模块导入警告，随后报错"Executing the custom container implementation failed"，表明NPU容器环境未正确初始化或配置，导致作业无法继续执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987706

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在执行过程中出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987719

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987755

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在执行过程中自定义容器实现失败，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987806

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在加载模型分片（约18%）时，自定义容器实现执行失败，导致任务中止。可能是容器环境或资源问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95189888948

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987547) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987643) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956957774/job/95188987746) |


## [Run #31956934352](https://github.com/sgl-project/sglang/actions/runs/31956934352)
- **分支**: `ling3-flash-dspark`
- **总耗时**: 113.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31956934352

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 38.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919229) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | 性能回归 | NPU性能测试用例失败，未达到预期性能指标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95189498041) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 29.4min | 性能回归 | NPU性能测试未达标，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95191732485) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95193782410) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95202475683) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最终仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919229

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，运行1130秒后退出码为1，属于性能测试未通过，可能因模型性能未达标或环境波动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95189498041

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: qwen3_235b_a22b模型的w8a8_8p_in3k5_out1k5_50ms性能测试用例返回退出码1，测试耗时1437秒，未达到预期性能指标，导致0/4测试全部失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95191732485

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，触发了fast-fail机制，本作业未实际执行测试即被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95193782410

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现其他作业（如multimodal-gen-test-1-npu-a3等）失败，本作业被标记为级联失败并跳过，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95202475683

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919149) |
| base-b-test-16-npu-a3 / run (0) | 52.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919173) |
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919182) |
| base-b-test-4-npu-a3 / run (0) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919207) |
| base-b-test-4-npu-a3 / run (1) | 13.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919210) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919258) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919295) |
| base-b-test-1-npu-a3 / run (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919316) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919338) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919350) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 110.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31956934352/job/95188919359) |


## [Run #31955324456](https://github.com/sgl-project/sglang/actions/runs/31955324456)
- **分支**: `cjc/unified-swa-branching`
- **总耗时**: 298.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31955324456

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 55.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036018) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.9min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185886842) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因同批次其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95189680806) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95198145406) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，仅显示上传diffusion-failures目录时无文件，无法判断测试是否通过或失败原因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036018

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1110秒后失败，退出码1，属于性能测试未通过，可能因吞吐或延迟未达预期。
  链接: https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185886842

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95189680806

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95198145406

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036120) |
| base-b-test-4-npu-a3 / run (0) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036139) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036144) |
| base-b-test-2-npu-a3 / run (0) | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036150) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036153) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036165) |
| base-b-test-16-npu-a3 / run (0) | 46.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036241) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036321) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036354) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036379) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95185036397) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 269.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955324456/job/95188400280) |


## [Run #31955085141](https://github.com/sgl-project/sglang/actions/runs/31955085141)
- **分支**: `flashinfer-sm90-mxfp4-fp8`
- **总耗时**: 85.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31955085141

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 51.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440827) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 75.5min | 精度回归 | NPU精度测试用例qwen3_5_9b_bf16_1p_gsm8k失败，0/3通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184441067) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 39.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95188063490) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、Node.js弃用警告及上传diffusion-failures工件（未找到文件）等常规信息，未展示多模态生成测试的具体执行结果或错误，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440827

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，耗时4318秒超过预估3600秒，所有3个精度测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184441067

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95188063490

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440907) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440913) |
| base-b-test-4-npu-a3 / run (1) | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440914) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440932) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440960) |
| base-b-test-16-npu-a3 / run (0) | 54.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440963) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184440970) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184441074) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184441080) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95184441215) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95185906888) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955085141/job/95190190904) |


## [Run #31955054550](https://github.com/sgl-project/sglang/actions/runs/31955054550)
- **分支**: `fix-qwen2-marlin-fp8-bias`
- **总耗时**: 89.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31955054550

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 52.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358019) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184951328) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95187603282) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 2.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95189003321) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95194966680) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358019

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184951328

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95187603282

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95189003321

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95194966680

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 54.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358025) |
| base-b-test-8-npu-a3 / run (0) | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358041) |
| base-b-test-2-npu-a3 / run (0) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358042) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358053) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358071) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358101) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358138) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358180) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358183) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358226) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31955054550/job/95184358227) |


## [Run #31953585741](https://github.com/sgl-project/sglang/actions/runs/31953585741)
- **分支**: `codex/diffusion-reuse-srt-clip`
- **总耗时**: 87.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31953585741

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741361) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95181208297) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95184960104) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95185607948) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95191101856) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741361

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95181208297

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95184960104

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95185607948

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95191101856

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741452) |
| base-b-test-16-npu-a3 / run (0) | 56.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741493) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741496) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741509) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741521) |
| base-b-test-1-npu-a3 / run (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741524) |
| base-b-test-4-npu-a3 / run (0) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741540) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741594) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741609) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741613) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31953585741/job/95180741620) |


## [Run #31952159941](https://github.com/sgl-project/sglang/actions/runs/31952159941)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 90.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31952159941

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240376) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177930547) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 43.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95180238867) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95187986979) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240376

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177930547

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95180238867

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95187986979

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240419) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240438) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240471) |
| base-b-test-1-npu-a3 / run (0) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240484) |
| base-b-test-4-npu-a3 / run (0) | 32.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240530) |
| base-b-test-16-npu-a3 / run (0) | 51.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240610) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240628) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240672) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240697) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240715) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95177240722) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31952159941/job/95181711936) |


## [Run #31951691163](https://github.com/sgl-project/sglang/actions/runs/31951691163)
- **分支**: `mmangkad/migrate-causal-conv1d-jit`
- **总耗时**: 93.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31951691163

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 41.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043602) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176907218) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 27.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95179168777) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95181301558) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95187087098) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043602

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176907218

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95179168777

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95181301558

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95187087098

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043661) |
| base-b-test-4-npu-a3 / run (1) | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043671) |
| base-b-test-16-npu-a3 / run (0) | 48.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043696) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043697) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043698) |
| base-b-test-2-npu-a3 / run (0) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043779) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043786) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043816) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043823) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176043836) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31951691163/job/95176044101) |


## [Run #31950203437](https://github.com/sgl-project/sglang/actions/runs/31950203437)
- **分支**: `codex/diffusion-reuse-srt-qwen-vision`
- **总耗时**: 176.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31950203437

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 48.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447827) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 149.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95175506075) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95177063579) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95185622424) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447827

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95175506075

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95177063579

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95185622424

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447816) |
| base-b-test-16-npu-a3 / run (0) | 56.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447837) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447869) |
| base-b-test-4-npu-a3 / run (1) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447907) |
| base-b-test-1-npu-a3 / run (0) | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447910) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447938) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172447999) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172448051) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172448059) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172448086) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172451684) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31950203437/job/95172952265) |


## [Run #31948344563](https://github.com/sgl-project/sglang/actions/runs/31948344563)
- **分支**: `main`
- **总耗时**: 44.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31948344563

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31948344563/job/95167903402) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31948344563/job/95167903402


## [Run #31947880093](https://github.com/sgl-project/sglang/actions/runs/31947880093)
- **分支**: `codex/diffusion-reuse-srt-clip`
- **总耗时**: 85.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31947880093

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166762999) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95167222474) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95171193823) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95171641837) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95176406933) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166762999

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95167222474

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95171193823

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95171641837

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95176406933

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166762995) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763001) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763009) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763014) |
| base-b-test-2-npu-a3 / run (0) | 19.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763032) |
| base-b-test-16-npu-a3 / run (0) | 54.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763071) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763086) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763135) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763144) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763168) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947880093/job/95166763213) |


## [Run #31947333589](https://github.com/sgl-project/sglang/actions/runs/31947333589)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 88.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31947333589

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413575) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165854012) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95168578248) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95169705703) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95175348768) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413575

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165854012

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95168578248

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95169705703

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95175348768

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 48.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413742) |
| base-b-test-2-npu-a3 / run (0) | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413783) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413787) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413797) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413879) |
| base-b-test-1-npu-a3 / run (0) | 25.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413901) |
| base-b-test-4-npu-a3 / run (0) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165413908) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165414106) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165414155) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165414160) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947333589/job/95165414202) |


## [Run #31947030532](https://github.com/sgl-project/sglang/actions/runs/31947030532)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 7.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31947030532

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668098) |
| base-b-test-16-npu-a3 / run (0) | 5.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668102) |
| base-b-test-2-npu-a3 / run (0) | 5.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668111) |
| base-b-test-1-npu-a3 / run (0) | 5.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668138) |
| base-b-test-8-npu-a3 / run (0) | 4.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668154) |
| base-b-test-4-npu-a3 / run (1) | 5.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668166) |
| base-b-test-4-npu-a3 / run (0) | 5.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668204) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668254) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668263) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668283) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668314) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668098

- **base-b-test-16-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668102

- **base-b-test-2-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668111

- **base-b-test-1-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668138

- **base-b-test-8-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668154

- **base-b-test-4-npu-a3 / run (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668166

- **base-b-test-4-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668204

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668254

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668263

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668283

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668314

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31947030532/job/95164668143) |


## [Run #31946659232](https://github.com/sgl-project/sglang/actions/runs/31946659232)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 8.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31946659232

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 6.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163739927) |
| base-b-test-1-npu-a3 / run (0) | 6.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163739957) |
| base-b-test-16-npu-a3 / run (0) | 7.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163739993) |
| base-b-test-4-npu-a3 / run (0) | 7.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740002) |
| multimodal-gen-test-1-npu-a3 | 6.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740009) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740023) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740037) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740049) |
| base-b-test-4-npu-a3 / run (1) | 7.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740054) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740140) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95164348537) |

- **base-b-test-2-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163739927

- **base-b-test-1-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163739957

- **base-b-test-16-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163739993

- **base-b-test-4-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740002

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740009

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740023

- **base-b-test-8-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740037

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740049

- **base-b-test-4-npu-a3 / run (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740054

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740140

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95164348537

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163739929) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946659232/job/95163740095) |


## [Run #31946576010](https://github.com/sgl-project/sglang/actions/runs/31946576010)
- **分支**: `main`
- **总耗时**: 39.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31946576010

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946576010/job/95163503135) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946576010/job/95163503135


## [Run #31946415467](https://github.com/sgl-project/sglang/actions/runs/31946415467)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 5.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31946415467

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102145) |
| base-a-test-1-npu-a2 / run (0) | 4.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102169) |
| base-b-test-16-npu-a3 / run (0) | 1.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102203) |
| base-b-test-1-npu-a3 / run (0) | 0.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102218) |
| base-b-test-4-npu-a3 / run (0) | 2.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102223) |
| base-b-test-2-npu-a3 / run (0) | 1.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102233) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102305) |
| base-b-test-4-npu-a3 / run (1) | 2.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102335) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102414) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102433) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102440) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 2.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102456) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102145

- **base-a-test-1-npu-a2 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102169

- **base-b-test-16-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102203

- **base-b-test-1-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102218

- **base-b-test-4-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102223

- **base-b-test-2-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102233

- **base-b-test-8-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102305

- **base-b-test-4-npu-a3 / run (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102335

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102414

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102433

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102440

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946415467/job/95163102456


## [Run #31946164889](https://github.com/sgl-project/sglang/actions/runs/31946164889)
- **分支**: `codex/diffusion-reuse-srt-clip`
- **总耗时**: 38.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31946164889

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397286) |
| base-b-test-16-npu-a3 / run (0) | 30.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397363) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 34.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397422) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 33.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397424) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95163528019) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95166224220) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397286

- **base-b-test-16-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397363

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397422

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397424

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95163528019

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95166224220

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397289) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397295) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397304) |
| base-b-test-2-npu-a3 / run (0) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397334) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397335) |
| base-b-test-4-npu-a3 / run (0) | 31.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397365) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 28.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397504) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946164889/job/95162397524) |


## [Run #31946133977](https://github.com/sgl-project/sglang/actions/runs/31946133977)
- **分支**: `main`
- **总耗时**: 10.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31946133977

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362545) |
| base-b-test-16-npu-a3 / run (0) | 5.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362570) |
| base-b-test-1-npu-a3 / run (0) | 7.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362582) |
| base-b-test-8-npu-a3 / run (0) | 4.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362603) |
| base-b-test-4-npu-a3 / run (0) | 6.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362623) |
| base-b-test-2-npu-a3 / run (0) | 7.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362659) |
| base-b-test-4-npu-a3 / run (1) | 6.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362714) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362867) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362877) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362929) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95163269942) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362545

- **base-b-test-16-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362570

- **base-b-test-1-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362582

- **base-b-test-8-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362603

- **base-b-test-4-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362623

- **base-b-test-2-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362659

- **base-b-test-4-npu-a3 / run (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362714

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362867

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362877

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362929

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95163269942

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362630) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946133977/job/95162362889) |


## [Run #31946070244](https://github.com/sgl-project/sglang/actions/runs/31946070244)
- **分支**: `codex/diffusion-reuse-srt-siglip`
- **总耗时**: 86.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31946070244

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 36.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154012) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162708882) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 57.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95164768874) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95166820673) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95171826790) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154012

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162708882

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95164768874

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95166820673

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95171826790

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154060) |
| base-b-test-4-npu-a3 / run (0) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154105) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154116) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154139) |
| base-b-test-2-npu-a3 / run (0) | 19.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154158) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154175) |
| base-b-test-16-npu-a3 / run (0) | 51.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154196) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154252) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154254) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154259) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31946070244/job/95162154324) |


## [Run #31945849046](https://github.com/sgl-project/sglang/actions/runs/31945849046)
- **分支**: `codex/tighten-nv-perf-baselines-20260816`
- **总耗时**: 41.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31945849046

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 41.0min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945849046/job/95161604995) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945849046/job/95161604995


## [Run #31945459138](https://github.com/sgl-project/sglang/actions/runs/31945459138)
- **分支**: `main`
- **总耗时**: 15.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31945459138

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701775) |
| base-b-test-2-npu-a3 / run (0) | 13.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701884) |
| base-b-test-4-npu-a3 / run (1) | 12.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701885) |
| base-b-test-16-npu-a3 / run (0) | 13.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701896) |
| base-b-test-4-npu-a3 / run (0) | 8.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701935) |
| base-b-test-1-npu-a3 / run (0) | 13.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701950) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.6min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160702082) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.4min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160702100) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.3min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160702254) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 9.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95161198977) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701775

- **base-b-test-2-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701884

- **base-b-test-4-npu-a3 / run (1)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701885

- **base-b-test-16-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701896

- **base-b-test-4-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701935

- **base-b-test-1-npu-a3 / run (0)**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701950

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160702082

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160702100

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160702254

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95161198977

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701866) |
| base-b-test-8-npu-a3 / run (0) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160701870) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31945459138/job/95160702136) |


## [Run #31942715216](https://github.com/sgl-project/sglang/actions/runs/31942715216)
- **分支**: `codex/diffusion-reuse-srt-qwen-vision`
- **总耗时**: 92.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31942715216

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 41.1min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138692) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154679962) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 26.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95156741411) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95158520257) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95164104557) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138692

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154679962

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95156741411

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95158520257

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95164104557

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138679) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138701) |
| base-b-test-16-npu-a3 / run (0) | 54.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138779) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138780) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138808) |
| base-b-test-4-npu-a3 / run (1) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138837) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138861) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138908) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138928) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154138929) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31942715216/job/95154139054) |


## [Run #31941258191](https://github.com/sgl-project/sglang/actions/runs/31941258191)
- **分支**: `add_mxfp4w4a4_quantization_for_npu`
- **总耗时**: 81.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31941258191

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.5min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190428287) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | AI调用失败 | 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429756) |

- **multimodal-gen-test-1-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190428287

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions
  链接: https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429756

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190428710) |
| base-b-test-16-npu-a3 / run (0) | 56.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190428860) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190428952) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429000) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429037) |
| base-b-test-4-npu-a3 / run (0) | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429042) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429067) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429185) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429457) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429464) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 132.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429528) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429732) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 77.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190429750) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 263.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941258191/job/95190430372) |


---
*Auto-generated by npu_pr_monitor.py*