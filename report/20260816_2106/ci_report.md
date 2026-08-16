# NPU CI 执行监控
**生成时间**: 2026-08-16 13:06 UTC
**分析 Run 数**: 74

---

## 📊 本次执行总结

- **成功 Job 数**: 408
- **失败 Run 数**: 74
- **成功 Job 平均耗时**: 26.4min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 272.7min | #31928447033 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95126425075) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 130.0min | #31935629376 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032850) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 127.2min | #31934702496 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731835) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 125.3min | #31925179201 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592463) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 121.3min | #31922519115 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635462) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 116.1min | #31938576525 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399531) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 114.9min | #31928447033 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616442) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.2min | #31933845963 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132676190) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 103.4min | #31927569122 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417630) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 101.1min | #31928566924 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120378081) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31928447033<br>[#34245 [BCG][6/N] Allow prefill breakable CUDA graph for the Kimi archs](https://github.com/sgl-project/sglang/pull/34245) | `fix/kimi-bcg-multimodal-allowlist` | 346.6min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31928447033) |
| #31935629376<br>[#34801 [PD] Preserve decode KV across retraction in HiCache](https://github.com/sgl-project/sglang/pull/34801) | `shiyang/pd-host-pool-retraction-backup` | 211.7min | 3 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31935629376) |
| #31934702496<br>[#30531 [DSA] Skip indexer KV cache for skip-topk layers](https://github.com/sgl-project/sglang/pull/30531) | `mmangkad/reland-30310` | 208.2min | 3 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31934702496) |
| #31935886036<br>[#34982 [misc] Rename shared-read boundary to shared-read ends and fix wrapper delegation](https://github.com/sgl-project/sglang/pull/34982) | `lsyin/shared-read-default-pre-replay` | 179.3min | 3 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31935886036) |
| #31928566924<br>[#35004 [Diffusion] Reuse SRT CLIP encoder blocks](https://github.com/sgl-project/sglang/pull/35004) | `codex/diffusion-reuse-srt-clip` | 157.3min | 4 | multimodal-gen-test-1-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31928566924) |
| #31927917567<br>[#34793 refactor(hicache): flatten L2 transfer execution](https://github.com/sgl-project/sglang/pull/34793) | `oss/l2-transfer-consolidation` | 155.2min | 4 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31927917567) |
| #31927156822<br>[#34962 [Quantization] Fix GPTQ scheme attachment broken by LinearBase.scheme default](https://github.com/sgl-project/sglang/pull/34962) | `mmangkad/fix-gptq-scheme-attach` | 147.2min | 4 | multimodal-gen-test-1-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927156822) |
| #31927569122<br>[#34519 fix(hicache): limit load-back pending to write-back](https://github.com/sgl-project/sglang/pull/34519) | `fix/hicache-component-scoped-load-back` | 137.7min | 6 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927569122) |
| #31927906274<br>[#35000 Support unified SWA page mapping in attention metadata](https://github.com/sgl-project/sglang/pull/35000) | `sync/52d1a85cf-unified-swa-page-map` | 135.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927906274) |
| #31925179201<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `codex/k3-mm-cache-k3` | 128.4min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31925179201) |
| #31922519115<br>[#30575 [AMD] Enable Fast Triton Sparse MLA backend](https://github.com/sgl-project/sglang/pull/30575) | `feat/triton-sparse-mla` | 126.3min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31922519115) |
| #31927523236<br>[#34996 Increase post-capture decode memory reserve](https://github.com/sgl-project/sglang/pull/34996) | `sync/c96e2b686-post-capture-reserve` | 126.2min | 4 | multimodal-gen-test-1-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927523236) |
| #31938576525<br>[#34994 Build Rust extensions on demand in source checkouts](https://github.com/sgl-project/sglang/pull/34994) | `rust-ext-on-demand-build` | 125.5min | 3 | multimodal-gen-test-1-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31938576525) |
| #31927191790<br>[#34953 [Perf] Restore the 16-token router GEMM threshold on SM10X](https://github.com/sgl-project/sglang/pull/34953) | `mmangkad/restore-router-gemm-threshold` | 113.7min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927191790) |
| #31933845963<br>[#34928 [diffusion][kernel] Accelerate Sana BCG with bit-exact conv post-processing](https://github.com/sgl-project/sglang/pull/34928) | `agent/optimize-sana-bcg-conv-post` | 109.5min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31933845963) |
| #31926215311<br>[#34991 vlm: streamline vision sdpa reshapes](https://github.com/sgl-project/sglang/pull/34991) | `codex/srt-vision-sdpa-reshape` | 95.6min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31926215311) |
| #31926709432<br>[#34474 [AMD] Qwen3.5: guard attn layers against empty DP-attention batch](https://github.com/sgl-project/sglang/pull/34474) | `main` | 95.5min | 6 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31926709432) |
| #31937939150<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `codex/diffusion-reuse-srt-qwen-vision` | 94.0min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31937939150) |
| #31930934132<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `codex/diffusion-reuse-srt-qwen-vision` | 92.6min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31930934132) |
| #31922959336<br>[#34558 [Bugfix] Preserve MXFP4 Triton weights in sharded state](https://github.com/sgl-project/sglang/pull/34558) | `fix-mxfp4-sharded-state` | 90.3min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31922959336) |
| #31941313044<br>[#34299 [KDA] Add zero-copy native prefill checkpoints and packed decode](https://github.com/sgl-project/sglang/pull/34299) | `codex/sglang-phase-a-admission-rebased-20260810` | 89.0min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31941313044) |
| #31922197965<br>[#33602 [AMD] [GLM5] Add opt-in PTPC FP8 projections on gfx950](https://github.com/sgl-project/sglang/pull/33602) | `RM/glm52-ptpc-fp8-proj` | 88.8min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31922197965) |
| #31926254363<br>[#34990 fix: support optional input buffer widths in prefill graphs](https://github.com/sgl-project/sglang/pull/34990) | `fix/prefill-input-embeds-width` | 87.7min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31926254363) |
| #31922891724<br>[#30715 [AMD] [GLM5] Fuse DSA indexer query Hadamard + FP8 quant into one Triton kernel (gfx950)](https://github.com/sgl-project/sglang/pull/30715) | `RM/glm52-fuse-hadamard-quant` | 86.5min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31922891724) |
| #31921817121<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `codex/k3-mm-cache-k3` | 86.0min | 4 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31921817121) |
| #31926257258<br>[#34992 [Diffusion] Reuse SRT SigLIP in Pi0.5](https://github.com/sgl-project/sglang/pull/34992) | `codex/pi05-reuse-srt-siglip` | 85.3min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31926257258) |
| #31938668367<br>[#34995 [VLM] Avoid synchronizing multimodal placeholder counts](https://github.com/sgl-project/sglang/pull/34995) | `sync/efb38a842-mm-placeholder-count` | 78.5min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31938668367) |
| #31933121612<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 75.3min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31933121612) |
| #31928029285<br>[#35002 Support model-defined prefill input embedding width](https://github.com/sgl-project/sglang/pull/35002) | `sync/c6998c515-model-input-embed-width` | 74.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31928029285) |
| #31925388557<br>[#33726 fix(bcg): preserve Qwen3-VL DeepStack inputs during replay](https://github.com/sgl-project/sglang/pull/33726) | `fix/bcg-deepstack-replay-slot` | 71.7min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31925388557) |
| #31927554425<br>[#34997 Fix world-size-one aliasing in MLP batch sync](https://github.com/sgl-project/sglang/pull/34997) | `sync/778b30ce0-world-size-one-aliasing` | 67.0min | 11 | base-b-test-16-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927554425) |
| #31930481070<br>[#30318 [NPU] Add mxfp4-w4a8 MOE Quantization Support for NPU](https://github.com/sgl-project/sglang/pull/30318) | `main` | 60.1min | 8 | base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31930481070) |
| #31927954354<br>[#35001 [Frontend] Apply request header overrides to chat completions](https://github.com/sgl-project/sglang/pull/35001) | `sync/637c210863-chat-header-overrides` | 59.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927954354) |
| #31921994026<br>[#34736 [Diffusion] Unify component residency controls](https://github.com/sgl-project/sglang/pull/34736) | `codex/component-residency-policy` | 51.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31921994026) |
| #31928950250<br>[#34817 [Diffusion] Speed up MiniMax-H3 VAE decode on 2×H100](https://github.com/sgl-project/sglang/pull/34817) | `codex/h3-vae-resident-perf` | 51.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31928950250) |
| #31933843556<br>[#34929 [diffusion] Enable breakable CUDA graphs for LTX-2.3](https://github.com/sgl-project/sglang/pull/34929) | `agent/enable-ltx23-breakable-cuda-graph` | 51.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31933843556) |
| #31927745761<br>[#34998 Add explicit EPLB balancedness reporting modes](https://github.com/sgl-project/sglang/pull/34998) | `sync/b1b1f1038-eplb-report-modes` | 50.0min | 11 | base-b-test-4-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927745761) |
| #31921029777<br>[#34558 [Bugfix] Preserve MXFP4 Triton weights in sharded state](https://github.com/sgl-project/sglang/pull/34558) | `fix-mxfp4-sharded-state` | 50.0min | 5 | base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31921029777) |
| #31927889918<br>[#34967 [MoE] Add FlashInfer SM90 MXFP4 W4A8 MoE](https://github.com/sgl-project/sglang/pull/34967) | `flashinfer-sm90-mxfp4-fp8` | 49.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-b-test-4-npu-a3 / run (1) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927889918) |
| #31941711998<br>[#35016 [diffusion] test: tighten NVIDIA perf baselines](https://github.com/sgl-project/sglang/pull/35016) | `codex/tighten-nv-perf-baselines-20260816` | 48.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31941711998) |
| #31933836587<br>[#34932 [diffusion] Accelerate Cosmos3 T2I QKNorm+RoPE](https://github.com/sgl-project/sglang/pull/34932) | `agent/optimize-cosmos3-t2i-qknorm-rope` | 47.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31933836587) |
| #31933841111<br>[#34930 [diffusion] Reuse bit-exact modulation fast path for LTX-2.3](https://github.com/sgl-project/sglang/pull/34930) | `agent/reuse-ltx2-lossless-modulate-fast-path` | 46.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31933841111) |
| #31938560971<br>[#34478 [Spec] Support output logprobs with DSpark](https://github.com/sgl-project/sglang/pull/34478) | `zhisbug/dspark-output-logprobs` | 46.1min | 8 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31938560971) |
| #31943411462<br>[#34991 vlm: streamline vision sdpa reshapes](https://github.com/sgl-project/sglang/pull/34991) | `main` | 45.8min | 7 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31943411462) |
| #31933839449<br>[#34931 [diffusion] Accelerate lossless Ideogram norm post-processing](https://github.com/sgl-project/sglang/pull/34931) | `agent/optimize-ideogram-lossless-norm-post` | 45.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31933839449) |
| #31938769946<br>[#34931 [diffusion] Accelerate lossless Ideogram norm post-processing](https://github.com/sgl-project/sglang/pull/34931) | `main` | 44.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31938769946) |
| #31927485489<br>[#34995 [VLM] Avoid synchronizing multimodal placeholder counts](https://github.com/sgl-project/sglang/pull/34995) | `sync/efb38a842-mm-placeholder-count` | 43.5min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31927485489) |
| #31926577809 | `codex/h3-vae-resident-perf` | 41.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31926577809) |
| #31924313612<br>[#34663 [Diffusion] Refresh docs, retire stale knobs, and fix nightly attribution](https://github.com/sgl-project/sglang/pull/34663) | `codex/diffusion-biweekly-review-20260813` | 40.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31924313612) |
| #31935377653<br>[#34986 [diffusion] feat: load quantized H3 text encoder checkpoints](https://github.com/sgl-project/sglang/pull/34986) | `codex/minimax-h3-text-encoder-quant` | 37.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31935377653) |
| #31937344068<br>[#35016 [diffusion] test: tighten NVIDIA perf baselines](https://github.com/sgl-project/sglang/pull/35016) | `codex/tighten-nv-perf-baselines-20260816` | 36.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31937344068) |
| #31934812680<br>[#34962 [Quantization] Fix GPTQ scheme attachment broken by LinearBase.scheme default](https://github.com/sgl-project/sglang/pull/34962) | `main` | 35.9min | 8 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31934812680) |
| #31944319886<br>[#35016 [diffusion] test: tighten NVIDIA perf baselines](https://github.com/sgl-project/sglang/pull/35016) | `codex/tighten-nv-perf-baselines-20260816` | 33.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31944319886) |
| #31934266502<br>[#34801 [PD] Preserve decode KV across retraction in HiCache](https://github.com/sgl-project/sglang/pull/34801) | `shiyang/pd-host-pool-retraction-backup` | 31.8min | 7 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31934266502) |
| #31922969081<br>[#30900 [AMD][Quantization][Bugfix] Fix bug related to fp8 max on gfx95x for per-token-group quant (ROCm)](https://github.com/sgl-project/sglang/pull/30900) | `main` | 31.1min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31922969081) |
| #31932908145<br>[#34509 [JIT Kernel] Migrate moe_topk_softmax from AOT to JIT](https://github.com/sgl-project/sglang/pull/34509) | `main` | 31.0min | 8 | base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31932908145) |
| #31924180496<br>[#34736 [Diffusion] Unify component residency controls](https://github.com/sgl-project/sglang/pull/34736) | `main` | 27.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31924180496) |
| #31922040941<br>[#30808 [AMD] [GLM5] Enable dense-MHA short-context prefill fallback on gfx950](https://github.com/sgl-project/sglang/pull/30808) | `main` | 24.5min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31922040941) |
| #31921154636<br>[#34949 [Diffusion] Route MiniMax H3 VAE attention through native backends](https://github.com/sgl-project/sglang/pull/34949) | `main` | 23.2min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31921154636) |
| #31932247281<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 21.5min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31932247281) |
| #31937206738<br>[#34994 Build Rust extensions on demand in source checkouts](https://github.com/sgl-project/sglang/pull/34994) | `rust-ext-on-demand-build` | 17.1min | 9 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31937206738) |
| #31925695428<br>[#31794 [AMD][Fix] Qwen3.5: guard zero-grid launch in fused_qk_gemma_rmsnorm(_with_gate) (HIP invalid configuration on idle DP rank)](https://github.com/sgl-project/sglang/pull/31794) | `main` | 15.2min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31925695428) |
| #31935098211<br>[#34982 [misc] Rename shared-read boundary to shared-read ends and fix wrapper delegation](https://github.com/sgl-project/sglang/pull/34982) | `lsyin/shared-read-default-pre-replay` | 14.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31935098211) |
| #31936689061<br>[#34994 Build Rust extensions on demand in source checkouts](https://github.com/sgl-project/sglang/pull/34994) | `rust-ext-on-demand-build` | 12.8min | 11 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31936689061) |
| #31925270179<br>[#34870 Fix swa eviction frontier for bigram keys](https://github.com/sgl-project/sglang/pull/34870) | `main` | 11.1min | 11 | base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31925270179) |
| #31937923934<br>[#34994 Build Rust extensions on demand in source checkouts](https://github.com/sgl-project/sglang/pull/34994) | `rust-ext-on-demand-build` | 10.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31937923934) |
| #31936390452<br>[#34994 Build Rust extensions on demand in source checkouts](https://github.com/sgl-project/sglang/pull/34994) | `rust-ext-on-demand-build` | 7.8min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31936390452) |
| #31924636061 | `codex/h3-vae-resident-perf` | 7.5min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31924636061) |
| #31934514340<br>[#34663 [Diffusion] Refresh docs, retire stale knobs, and fix nightly attribution](https://github.com/sgl-project/sglang/pull/34663) | `main` | 7.2min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31934514340) |
| #31934954640<br>[#34982 [misc] Rename shared-read boundary to shared-read ends and fix wrapper delegation](https://github.com/sgl-project/sglang/pull/34982) | `lsyin/shared-read-default-pre-replay` | 6.8min | 10 | base-b-test-8-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31934954640) |
| #31926286150<br>[#33998 [HiCache] Optimize LogicalHostPool free-list release](https://github.com/sgl-project/sglang/pull/33998) | `main` | 6.6min | 12 | base-a-test-1-npu-a2 / run (0), base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31926286150) |
| #31938349305<br>[#34994 Build Rust extensions on demand in source checkouts](https://github.com/sgl-project/sglang/pull/34994) | `rust-ext-on-demand-build` | 6.0min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31938349305) |
| #31934185547<br>[#34793 refactor(hicache): flatten L2 transfer execution](https://github.com/sgl-project/sglang/pull/34793) | `main` | 5.2min | 11 | base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31934185547) |
| #31926035114 | `fix/prefill-input-embeds-width` | 5.1min | 10 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31926035114) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 57 | 48 | 0 | 9 |
| base-b-test-1-npu-a3 / run (0) | 57 | 31 | 0 | 26 |
| base-b-test-16-npu-a3 / run (0) | 56 | 28 | 1 | 27 |
| base-b-test-2-npu-a3 / run (0) | 56 | 35 | 0 | 21 |
| base-b-test-4-npu-a3 / run (0) | 57 | 28 | 0 | 29 |
| base-b-test-4-npu-a3 / run (1) | 57 | 38 | 0 | 19 |
| base-b-test-8-npu-a3 / run (0) | 57 | 41 | 0 | 16 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 56 | 36 | 1 | 19 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 57 | 24 | 0 | 33 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 57 | 32 | 0 | 25 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 56 | 41 | 0 | 15 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 36 | 1 | 0 | 35 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 24 | 5 | 0 | 19 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 32 | 9 | 0 | 23 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 40 | 5 | 0 | 35 |
| multimodal-gen-test-1-npu-a3 | 71 | 6 | 45 | 20 |

---


## [Run #31944319886](https://github.com/sgl-project/sglang/actions/runs/31944319886)
- **分支**: `codex/tighten-nv-perf-baselines-20260816`
- **总耗时**: 33.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31944319886

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 33.0min | 环境问题 | 作业因环境问题失败，未找到失败产物文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31944319886/job/95157962442) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业运行约32分钟后结束，上传diffusion-failures产物时提示未找到文件，说明测试未产生失败样本，可能因环境配置或依赖问题导致测试未正常执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31944319886/job/95157962442


## [Run #31943411462](https://github.com/sgl-project/sglang/actions/runs/31943411462)
- **分支**: `main`
- **总耗时**: 45.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31943411462

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 36.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790010) |
| base-b-test-4-npu-a3 / run (0) | 8.5min | 环境问题 | NPU测试用例test_npu_hicache_mla.py执行失败，返回退出码1，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790092) |
| base-b-test-16-npu-a3 / run (0) | 41.6min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790096) |
| base-b-test-1-npu-a3 / run (0) | 44.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790154) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 44.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790311) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95158746629) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95160162917) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790010

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test/registered/npu/basic_function/HiCache/test_npu_hicache_mla.py在运行约283秒后失败，退出码为1，最终导致整个作业失败。日志中未显示具体错误原因，但可能是环境配置或测试用例本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790092

- **base-b-test-16-npu-a3 / run (0)**: 作业在加载模型分片（约23%）时，自定义容器实现执行失败，提示联系自托管runner管理员，可能因容器环境或资源问题导致中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790096

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790154

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790311

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-b-test-4-npu-a3 作业失败，根因失败作业被过滤后触发 fast-fail，导致本作业未实际运行即被终止，属于上游作业失败引发的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95158746629

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动前的PR健康检查阶段，检测到multimodal-gen-test-1-npu-a3和base-b-test-4-npu-a3两个根因作业失败，因此本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95160162917

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790090) |
| base-b-test-2-npu-a3 / run (0) | 28.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790109) |
| base-b-test-8-npu-a3 / run (0) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790110) |
| base-b-test-4-npu-a3 / run (1) | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790169) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790307) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790336) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95155790363) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31943411462/job/95156234009) |


## [Run #31941711998](https://github.com/sgl-project/sglang/actions/runs/31941711998)
- **分支**: `codex/tighten-nv-perf-baselines-20260816`
- **总耗时**: 48.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31941711998

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 47.8min | 日志下载失败 | HTTPSConnectionPool(host='productionresultssa17.blob.core.windows.net', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/31941711998/job/95151802200) |



## [Run #31941313044](https://github.com/sgl-project/sglang/actions/runs/31941313044)
- **分支**: `codex/sglang-phase-a-admission-rebased-20260810`
- **总耗时**: 89.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31941313044

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 41.0min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150861872) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.8min | 性能回归 | NPU性能测试未达预期，测试用例失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95151938926) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95154120980) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95155302727) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 环境问题 | PR健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95160064745) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150861872

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1087秒后失败，返回码1，可能因性能指标未达标或执行错误导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95151938926

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到根因失败作业base-c-test-perf-8-npu-a3，触发Fast-fail跳过本作业，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95154120980

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95155302727

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3作业失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95160064745

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 46.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150861926) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150861996) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862051) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862078) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862099) |
| base-b-test-2-npu-a3 / run (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862100) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862111) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862222) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862240) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862288) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31941313044/job/95150862347) |


## [Run #31938769946](https://github.com/sgl-project/sglang/actions/runs/31938769946)
- **分支**: `main`
- **总耗时**: 44.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31938769946

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 44.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938769946/job/95144802969) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js版本弃用警告和上传artifact时未找到失败文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938769946/job/95144802969


## [Run #31938668367](https://github.com/sgl-project/sglang/actions/runs/31938668367)
- **分支**: `sync/efb38a842-mm-placeholder-count`
- **总耗时**: 78.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31938668367

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.9min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545305) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.9min | 其他 | 健康检查检测到其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545466) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.8min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95145496355) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 45.9min | 性能回归 | qwen3_235b_a22b性能测试未通过，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95147857131) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因健康检查发现其他根因作业失败而被跳过，并非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95148859271) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业正常执行，最后上传diffusion-failures目录时提示无文件，未发现测试失败或错误信息，可能为作业提前结束或测试未产生失败产物。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545305

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段被PR健康检查快速失败机制终止，原因是同一次运行中multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个作业已失败，本作业被级联跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545466

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1082秒后退出码为1，属于性能测试未通过，可能因推理延迟或吞吐量未达预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95145496355

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中qwen3_235b_a22b测试失败（exit code 1），而其他两个测试通过。该测试耗时1436秒，可能未达到性能目标或出现错误，需查看具体日志确认是性能不达标还是运行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95147857131

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到另一个作业base-c-test-perf-8-npu-a3失败，触发了快速失败机制，导致本作业被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95148859271

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545321) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545350) |
| base-b-test-4-npu-a3 / run (1) | 12.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545356) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545360) |
| base-b-test-16-npu-a3 / run (0) | 52.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545364) |
| base-b-test-2-npu-a3 / run (0) | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545366) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545400) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545520) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545531) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938668367/job/95144545539) |


## [Run #31938576525](https://github.com/sgl-project/sglang/actions/runs/31938576525)
- **分支**: `rust-ext-on-demand-build`
- **总耗时**: 125.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31938576525

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399304) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 27.4min | 性能回归 | NPU性能测试未通过，0/4测试全部失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95147098480) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95157273250) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399304

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中4个性能测试全部失败，首个失败用例为qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms测试，运行1395秒后退出码1，可能因性能未达预期或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95147098480

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现其他两个作业（multimodal-gen-test-1-npu-a3和base-c-test-perf-16-npu-a3）失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95157273250

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399326) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399359) |
| base-b-test-4-npu-a3 / run (1) | 13.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399372) |
| base-b-test-16-npu-a3 / run (0) | 52.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399374) |
| base-b-test-4-npu-a3 / run (0) | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399378) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399395) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399396) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399465) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399481) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399526) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 116.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95144399531) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95145223412) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938576525/job/95148353747) |


## [Run #31938560971](https://github.com/sgl-project/sglang/actions/runs/31938560971)
- **分支**: `zhisbug/dspark-output-logprobs`
- **总耗时**: 46.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31938560971

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 45.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270559) |
| base-b-test-16-npu-a3 / run (0) | 1.5min | 环境问题 | Kubernetes Pod 启动失败，运行环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270630) |
| base-b-test-4-npu-a3 / run (0) | 20.0min | 代码错误 | NPU DP注意力测试失败，导致作业整体失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270636) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270676) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270823) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270827) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144721821) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95148022211) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures工件时未找到文件，说明测试可能未产生失败样本，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270559

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段即失败，日志显示 Pod 状态为 Failed，无法正常上线。这是自托管 runner 的容器调度问题，与代码或测试内容无关，属于基础设施环境故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270630

- **base-b-test-4-npu-a3 / run (0)**: test_npu_dp_attention.py 测试返回退出码1，5个测试中仅1个通过，该测试耗时783秒后失败，具体错误信息未在日志中显示，但可确定是测试用例本身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270636

- **base-b-test-1-npu-a3 / run (0)**: 本作业在启动前的健康检查中检测到根因作业 base-b-test-16-npu-a3 / run (0) 已失败，触发了 fast-fail 机制，因此本作业未实际运行测试即被跳过并报错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270676

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业 base-b-test-16-npu-a3 / run (0) 失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270823

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-16-npu-a3/run(0)，触发fast-fail机制，本作业未实际运行即被跳过，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270827

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到base-b-test-16-npu-a3作业失败，将其视为根因，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144721821

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-b-test-16-npu-a3和base-b-test-4-npu-a3两个根因作业失败，本作业作为级联失败被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95148022211

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270600) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270640) |
| base-b-test-2-npu-a3 / run (0) | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270642) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270720) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270796) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31938560971/job/95144270811) |


## [Run #31938349305](https://github.com/sgl-project/sglang/actions/runs/31938349305)
- **分支**: `rust-ext-on-demand-build`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31938349305

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 1.4min | 环境问题 | 作业在准备阶段即失败，未进入实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798410) |
| base-b-test-2-npu-a3 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798498) |
| base-b-test-8-npu-a3 / run (0) | 4.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798511) |
| base-b-test-1-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798522) |
| base-a-test-1-npu-a2 / run (0) | 4.1min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798535) |
| base-b-test-4-npu-a3 / run (1) | 3.2min | 环境问题 | 自定义容器执行失败，导致作业在安装依赖后崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798565) |
| base-b-test-4-npu-a3 / run (0) | 4.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化或模型加载过程中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798567) |
| base-b-test-16-npu-a3 / run (0) | 2.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798570) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.7min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798651) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.9min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798682) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.8min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798683) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798702) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在checkout后立即结束，仅有Node.js 20弃用警告，无测试执行或错误信息，疑似基础设施或调度问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798410

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型权重加载过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798498

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798511

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798522

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行测试前执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798535

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在成功安装sglang_router后，出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798565

- **base-b-test-4-npu-a3 / run (0)**: 日志显示模型加载到多线程分片阶段时，自定义容器实现执行失败，可能是NPU驱动、容器配置或资源分配问题，导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798567

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在安装custom_ops后，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798570

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在设置环境变量后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，属于自托管runner环境或容器配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798651

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），属于自托管runner环境问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798682

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798683

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型加载过程中（约3%分片时）出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于容器环境异常，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31938349305/job/95143798702


## [Run #31937939150](https://github.com/sgl-project/sglang/actions/runs/31937939150)
- **分支**: `codex/diffusion-reuse-srt-qwen-vision`
- **总耗时**: 94.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31937939150

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 51.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746332) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.3min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95143962283) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.4min | 其他 | 健康检查发现同PR中其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95145948864) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95147252339) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95152437326) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746332

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1068秒后退出码1，0/1通过，属于性能测试未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95143962283

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到同PR的base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95145948864

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 本作业在启动前的PR健康检查中发现根因失败作业base-c-test-perf-8-npu-a3，触发快速失败机制，未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95147252339

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在“Check PR test health”步骤被跳过，原因是根因作业multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3失败，触发了fast-fail机制，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95152437326

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 52.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746344) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746350) |
| base-b-test-2-npu-a3 / run (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746352) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746365) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746401) |
| base-b-test-1-npu-a3 / run (0) | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746425) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746442) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746525) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746535) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746549) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937939150/job/95142746661) |


## [Run #31937923934](https://github.com/sgl-project/sglang/actions/runs/31937923934)
- **分支**: `rust-ext-on-demand-build`
- **总耗时**: 10.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31937923934

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.0min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746607) |
| base-b-test-2-npu-a3 / run (0) | 9.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746613) |
| base-b-test-8-npu-a3 / run (0) | 1.9min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746616) |
| base-b-test-1-npu-a3 / run (0) | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746623) |
| base-b-test-16-npu-a3 / run (0) | 5.2min | 环境问题 | 自托管runner容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746631) |
| base-b-test-4-npu-a3 / run (1) | 8.8min | 环境问题 | 测试通过但自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746644) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746687) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746733) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.0min | 环境问题 | 自定义容器执行失败，导致测试未运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746820) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746857) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746861) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体信息，仅有Node.js版本弃用警告和上传artifact时未找到失败文件。可能因日志截断或作业在测试前被取消，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746607

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的工件/缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746613

- **base-b-test-8-npu-a3 / run (0)**: 作业在安装torch-npu后执行自定义容器时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746616

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746623

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型权重分片时（6%进度），自定义容器实现执行失败，提示联系runner管理员，属于NPU自托管环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746631

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试用例TestAscendW4A4.test_gsm8k运行成功（OK），但随后出现"Executing the custom container implementation failed"错误，属于自托管runner容器执行环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746644

- **base-b-test-4-npu-a3 / run (0)**: 日志显示torch_npu的transfer_to_npu模块在容器启动时产生ImportWarning和RuntimeWarning，随后自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746687

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在加载模型分片时，GitHub Actions 报错“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746733

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装evalscope后，执行测试脚本时出现'Executing the custom container implementation failed'错误，属于自托管runner环境问题，而非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746820

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746857

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746861

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937923934/job/95142746658) |


## [Run #31937344068](https://github.com/sgl-project/sglang/actions/runs/31937344068)
- **分支**: `codex/tighten-nv-perf-baselines-20260816`
- **总耗时**: 36.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31937344068

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 35.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937344068/job/95141267164) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未出现测试执行或失败的具体错误，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937344068/job/95141267164


## [Run #31937206738](https://github.com/sgl-project/sglang/actions/runs/31937206738)
- **分支**: `rust-ext-on-demand-build`
- **总耗时**: 17.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31937206738

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.4min | 环境问题 | 作业因未找到diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140984956) |
| base-b-test-16-npu-a3 / run (0) | 16.1min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985044) |
| base-b-test-4-npu-a3 / run (0) | 15.6min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985055) |
| base-b-test-1-npu-a3 / run (0) | 14.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985120) |
| base-b-test-2-npu-a3 / run (0) | 15.3min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985137) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 15.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985203) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985223) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 15.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985276) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 15.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985291) |

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤未找到diffusion-failures/目录，说明测试未生成失败样本，作业可能因前置条件未满足或测试未运行而失败，属于环境或流程配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140984956

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型分片过程中，自定义容器实现执行失败（Executing the custom container implementation failed），导致作业中断。这属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985044

- **base-b-test-4-npu-a3 / run (0)**: 日志显示服务已成功启动，但随后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985055

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载完成后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985120

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在模型加载和tokenizer初始化后，出现'Executing the custom container implementation failed'错误，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985137

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985203

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但最后报错“Executing the custom container implementation failed”，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985223

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，提示联系runner管理员，属于NPU自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985276

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在09:00:56时出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985291

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985028) |
| base-a-test-1-npu-a2 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985030) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31937206738/job/95140985037) |


## [Run #31936689061](https://github.com/sgl-project/sglang/actions/runs/31936689061)
- **分支**: `rust-ext-on-demand-build`
- **总耗时**: 12.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31936689061

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719630) |
| base-a-test-1-npu-a2 / run (0) | 1.9min | 环境问题 | GitHub API 请求失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719729) |
| base-b-test-16-npu-a3 / run (0) | 10.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719797) |
| base-b-test-2-npu-a3 / run (0) | 10.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719807) |
| base-b-test-1-npu-a3 / run (0) | 10.7min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719811) |
| base-b-test-4-npu-a3 / run (1) | 11.1min | 环境问题 | 自定义容器执行失败，NPU后端不支持aten::_assert_async算子导致服务异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719839) |
| base-b-test-4-npu-a3 / run (0) | 10.8min | 环境问题 | 自定义容器执行失败，测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719860) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719987) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139720019) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139720020) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139720045) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719630

- **base-a-test-1-npu-a2 / run (0)**: github-script 调用 GitHub API 查询 lint 检查状态时返回 500 错误（fetch failed），属于 GitHub 服务端临时故障或网络问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719729

- **base-b-test-16-npu-a3 / run (0)**: 日志显示服务启动正常，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719797

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试本身通过（4 tests OK），但在运行下一个测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719807

- **base-b-test-1-npu-a3 / run (0)**: 测试运行到第4个文件时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719811

- **base-b-test-4-npu-a3 / run (1)**: 日志显示NPU后端不支持aten::_assert_async算子，回退到CPU执行，随后health_generate接口返回503，最终自定义容器执行失败，属于NPU环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719839

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试正在运行（TestDPAttentionDP2TP2.test_regex_generate_phone），但随后出现"Executing the custom container implementation failed"错误，说明自托管runner的容器环境出现问题，导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719860

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在评估gsm8k过程中，容器执行报错“Executing the custom container implementation failed”，随后作业终止，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719987

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139720019

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中容器突然终止，报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139720020

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139720045

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31936689061/job/95139719667) |


## [Run #31936390452](https://github.com/sgl-project/sglang/actions/runs/31936390452)
- **分支**: `rust-ext-on-demand-build`
- **总耗时**: 7.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31936390452

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929668) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929727) |
| base-a-test-1-npu-a2 / run (0) | 6.5min | 环境问题 | 自定义容器执行失败，NPU测试环境启动异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929733) |
| base-b-test-2-npu-a3 / run (0) | 6.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929749) |
| base-b-test-4-npu-a3 / run (1) | 6.4min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929765) |
| base-b-test-1-npu-a3 / run (0) | 6.5min | 环境问题 | 自定义容器执行失败，导致测试中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929786) |
| base-b-test-16-npu-a3 / run (0) | 6.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929845) |
| base-b-test-4-npu-a3 / run (0) | 6.4min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929873) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.4min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929926) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.5min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929940) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | 环境问题 | 测试套件未找到任何测试用例，但容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929954) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929967) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理过程，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929668

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929727

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行测试前执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929733

- **base-b-test-2-npu-a3 / run (0)**: 测试在生成请求后，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929749

- **base-b-test-4-npu-a3 / run (1)**: 测试在运行第二个文件时，自定义容器实现执行失败，错误信息提示联系自托管runner管理员，属于环境或基础设施问题，而非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929765

- **base-b-test-1-npu-a3 / run (0)**: 测试运行到第二个文件时，自定义容器实现执行失败，报错提示联系自托管runner管理员，属于环境或基础设施问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929786

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是因为日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929845

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试在捕获批次时（bs=12）出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929873

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在加载模型分片（约77%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929926

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在安装evalscope等依赖后，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929940

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示测试套件base-c-test-acc-8-npu-a3未找到匹配的测试用例（No tests found），属于预期跳过，但随后容器实现执行失败（Executing the custom container implementation failed），可能是自托管runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929954

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在加载模型分片时，自定义容器实现执行失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31936390452/job/95138929967


## [Run #31935886036](https://github.com/sgl-project/sglang/actions/runs/31935886036)
- **分支**: `lsyin/shared-read-default-pre-replay`
- **总耗时**: 179.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31935886036

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 35.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137711972) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达预期，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95138124709) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 27.5min | 性能回归 | NPU性能测试未通过，4个测试全部失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95140820096) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137711972

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1105秒，未通过性能基准要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95138124709

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行1421秒后失败，退出码1，所有4个性能测试均未通过，可能因性能未达预期或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95140820096

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137711967) |
| base-b-test-16-npu-a3 / run (0) | 50.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137711985) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137711989) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137711995) |
| base-a-test-1-npu-a2 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137711997) |
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137712017) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137712047) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 97.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137712100) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137712106) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137712160) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95137712172) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95142051909) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 79.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935886036/job/95148057422) |


## [Run #31935629376](https://github.com/sgl-project/sglang/actions/runs/31935629376)
- **分支**: `shiyang/pd-host-pool-retraction-backup`
- **总耗时**: 211.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31935629376

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032589) |
| base-b-test-16-npu-a3 / run (0) | 42.7min | 代码错误 | NPU PD分离测试用例失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032748) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 46.9min | 性能回归 | qwen3_235b_a22b性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95139788608) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行过程或失败信息，仅显示上传diffusion-failures工件时未找到文件，可能测试未运行或日志被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032589

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（exit code 1），其余3个测试通过。该测试耗时835秒，可能涉及PD分离功能逻辑错误或环境配置问题，需查看具体断言或错误日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032748

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试中qwen3_235b_a22b_w8a8_8p_in3k5_out1k5_50ms.py退出码为1，耗时1446秒，未通过性能阈值，其他两个测试通过，判定为性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95139788608

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032641) |
| base-b-test-2-npu-a3 / run (0) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032653) |
| base-b-test-4-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032663) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032672) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032695) |
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032707) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 130.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032850) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032880) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032890) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137032900) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95137632556) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95141180597) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 80.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935629376/job/95150790296) |


## [Run #31935377653](https://github.com/sgl-project/sglang/actions/runs/31935377653)
- **分支**: `codex/minimax-h3-text-encoder-quant`
- **总耗时**: 37.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31935377653

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 36.9min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935377653/job/95141134198) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时无文件，可能测试通过或未生成失败产物，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935377653/job/95141134198


## [Run #31935098211](https://github.com/sgl-project/sglang/actions/runs/31935098211)
- **分支**: `lsyin/shared-read-default-pre-replay`
- **总耗时**: 14.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31935098211

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030203) |
| base-b-test-16-npu-a3 / run (0) | 9.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030292) |
| base-b-test-8-npu-a3 / run (0) | 3.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030293) |
| base-b-test-4-npu-a3 / run (1) | 10.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030304) |
| base-b-test-1-npu-a3 / run (0) | 9.1min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030333) |
| base-b-test-2-npu-a3 / run (0) | 8.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030366) |
| base-b-test-4-npu-a3 / run (0) | 10.0min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030387) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030409) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.0min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030415) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030481) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.1min | 环境问题 | 自定义容器启动失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136905740) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030203

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030292

- **base-b-test-8-npu-a3 / run (0)**: 作业在运行自定义容器实现时失败，错误提示'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030293

- **base-b-test-4-npu-a3 / run (1)**: 作业在加载模型权重时（约69%进度）自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030304

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行到第3个用例时，自定义容器实现执行失败，导致作业终止。错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030333

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型加载到81%时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030366

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试运行正常，但随后出现错误：Executing the custom container implementation failed. Please contact your self hosted runner administrator. 这属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030387

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在加载模型分片时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，而非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030409

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在08:08:22出现错误“Executing the custom container implementation failed”，提示联系runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030415

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中NPU推理正常，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030481

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在安装依赖包时，GitHub Actions 执行自定义容器实现失败，报错 'Executing the custom container implementation failed'，属于自托管运行器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136905740

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030335) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31935098211/job/95136030427) |


## [Run #31934954640](https://github.com/sgl-project/sglang/actions/runs/31934954640)
- **分支**: `lsyin/shared-read-default-pre-replay`
- **总耗时**: 6.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31934954640

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 1.1min | 环境问题 | 自定义容器启动失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358019) |
| multimodal-gen-test-1-npu-a3 | 1.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358023) |
| base-b-test-4-npu-a3 / run (0) | 1.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358034) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358044) |
| base-b-test-4-npu-a3 / run (1) | 1.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358060) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358066) |
| base-b-test-1-npu-a3 / run (0) | 5.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358146) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358198) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358201) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358203) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示在安装系统依赖包（如libgl1、libicu-dev等）过程中，执行自定义容器实现失败，报错提示联系自托管runner管理员，属于运行环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358019

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358023

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358034

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358044

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储对象缺失，可能是文件被删除、路径错误或上传未完成，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358060

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在apt更新后执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358066

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358146

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358198

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358201

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934954640/job/95135358203


## [Run #31934812680](https://github.com/sgl-project/sglang/actions/runs/31934812680)
- **分支**: `main`
- **总耗时**: 35.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31934812680

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 28.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023718) |
| base-b-test-4-npu-a3 / run (0) | 9.0min | 环境问题 | NPU测试用例test_npu_hicache_mla.py执行失败，返回退出码1，导致整个作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023754) |
| base-b-test-1-npu-a3 / run (0) | 34.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023763) |
| base-b-test-16-npu-a3 / run (0) | 30.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023769) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 33.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023978) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 33.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023999) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135654169) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.3min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95138387763) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现测试执行或失败的具体错误信息，可能日志被截断或作业在测试前已终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023718

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test/registered/npu/basic_function/HiCache/test_npu_hicache_mla.py在运行281秒后失败，退出码为1。日志未显示具体错误原因，但测试摘要显示0/5通过，可能是环境配置或依赖问题导致测试无法正常运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023754

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023763

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型分片过程中，自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU容器环境配置问题或资源限制导致，并非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023769

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后报错"Executing the custom container implementation failed"，提示联系runner管理员，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023978

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（HTTP 200），但随后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023999

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135654169

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示执行自定义容器实现时失败（Executing the custom container implementation failed），提示联系自托管 runner 管理员，属于 NPU CI 基础设施或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95138387763

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023781) |
| base-b-test-2-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023789) |
| base-b-test-4-npu-a3 / run (1) | 27.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023790) |
| base-b-test-8-npu-a3 / run (0) | 11.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023809) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135023958) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934812680/job/95135024030) |


## [Run #31934702496](https://github.com/sgl-project/sglang/actions/runs/31934702496)
- **分支**: `mmangkad/reland-30310`
- **总耗时**: 208.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31934702496

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 36.6min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731649) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未通过，minimax_m2_5测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95135424463) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 59.9min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，疑似性能未达标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95139848708) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未包含实际测试命令输出或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731649

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1129秒。该测试为性能测试，可能未达到预期性能指标（如50ms延迟要求），导致测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95135424463

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例退出码1，其他用例通过，判断为该用例性能指标未满足要求，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95139848708

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731673) |
| base-b-test-16-npu-a3 / run (0) | 61.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731690) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731691) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731693) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731706) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731709) |
| base-b-test-4-npu-a3 / run (1) | 13.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731752) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731819) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 127.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731835) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731851) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95134731898) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95139187735) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 77.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934702496/job/95148406926) |


## [Run #31934514340](https://github.com/sgl-project/sglang/actions/runs/31934514340)
- **分支**: `main`
- **总耗时**: 7.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31934514340

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934514340/job/95134315047) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出。可能因日志截断或作业在测试前被取消，需查看完整日志以确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934514340/job/95134315047


## [Run #31934266502](https://github.com/sgl-project/sglang/actions/runs/31934266502)
- **分支**: `shiyang/pd-host-pool-retraction-backup`
- **总耗时**: 31.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31934266502

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708402) |
| base-b-test-16-npu-a3 / run (0) | 28.6min | 环境问题 | 自托管runner执行自定义容器实现失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708470) |
| base-b-test-4-npu-a3 / run (0) | 28.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708471) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 28.0min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708680) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708705) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95134619225) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.9min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95136371638) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708402

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载完成后，runner报错"Executing the custom container implementation failed"，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708470

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载模型权重后，执行自定义容器时失败，报错提示联系自托管runner管理员，属于NPU容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708471

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行约27分钟后，在正常解码阶段突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708680

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但随后出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708705

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示性能测试正常进行，但在运行过程中出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95134619225

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示在安装依赖后，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95136371638

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708426) |
| base-b-test-8-npu-a3 / run (0) | 8.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708429) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708431) |
| base-b-test-2-npu-a3 / run (0) | 19.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708472) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708574) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708636) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31934266502/job/95133708764) |


## [Run #31934185547](https://github.com/sgl-project/sglang/actions/runs/31934185547)
- **分支**: `main`
- **总耗时**: 5.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31934185547

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 2.5min | 环境问题 | 自定义容器执行失败，导致作业在构建阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531252) |
| base-b-test-4-npu-a3 / run (0) | 2.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531253) |
| base-a-test-1-npu-a2 / run (0) | 3.9min | 环境问题 | 自定义容器执行失败，NPU CI 环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531260) |
| multimodal-gen-test-1-npu-a3 | 3.6min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531265) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531269) |
| base-b-test-2-npu-a3 / run (0) | 3.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531271) |
| base-b-test-1-npu-a3 / run (0) | 3.1min | 环境问题 | 自定义容器执行失败，导致作业在启动阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531279) |
| base-b-test-16-npu-a3 / run (0) | 4.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531316) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.7min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531355) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.4min | 环境问题 | 自定义容器启动失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531387) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.5min | 环境问题 | 自定义容器执行失败，导致作业在启动阶段即中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531459) |

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在创建PEP 517构建环境时，执行自定义容器实现失败，错误信息为'Executing the custom container implementation failed'，可能是自托管runner环境配置或容器镜像问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531252

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531253

- **base-a-test-1-npu-a2 / run (0)**: 作业在安装依赖时，自定义容器实现执行失败，提示联系自托管 runner 管理员，属于 NPU CI 基础设施环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531260

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal测试的实际执行输出或失败原因，可能因日志截断或作业在测试前被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531265

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于基础设施或容器配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531269

- **base-b-test-2-npu-a3 / run (0)**: 作业在运行测试前，执行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531271

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在准备Python环境后，执行自定义容器实现时失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531279

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失或配置错误，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531316

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示执行自定义容器实现时失败，提示联系自托管 runner 管理员，可能是容器镜像或运行环境配置问题，非测试代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531355

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示在安装rustup组件时，自定义容器实现执行失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531387

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在运行自定义容器实现时出现错误（Executing the custom container implementation failed），随后作业进入清理阶段，未执行任何测试。这属于自托管运行器环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31934185547/job/95133531459


## [Run #31933845963](https://github.com/sgl-project/sglang/actions/runs/31933845963)
- **分支**: `agent/optimize-sana-bcg-conv-post`
- **总耗时**: 109.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31933845963

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 54.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132675835) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95133197122) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95135543435) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95136827936) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95143748738) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业整体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132675835

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1128秒后失败，0/1测试通过，属于性能测试未达标，可能因模型推理速度或吞吐量低于预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95133197122

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95135543435

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95136827936

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业（multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3），触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95143748738

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132675898) |
| base-b-test-2-npu-a3 / run (0) | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132675921) |
| base-b-test-16-npu-a3 / run (0) | 51.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132675925) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132675948) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132675953) |
| base-b-test-4-npu-a3 / run (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132676022) |
| base-b-test-4-npu-a3 / run (0) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132676030) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132676099) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132676120) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132676128) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933845963/job/95132676190) |


## [Run #31933843556](https://github.com/sgl-project/sglang/actions/runs/31933843556)
- **分支**: `agent/enable-ltx23-breakable-cuda-graph`
- **总耗时**: 51.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31933843556

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 50.5min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933843556/job/95132668431) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传工件步骤，且未找到diffusion-failures文件。中间关键测试输出被省略，无法判断具体失败原因，可能为测试未执行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933843556/job/95132668431


## [Run #31933841111](https://github.com/sgl-project/sglang/actions/runs/31933841111)
- **分支**: `agent/reuse-ltx2-lossless-modulate-fast-path`
- **总耗时**: 46.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31933841111

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 45.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933841111/job/95132662347) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含多模态生成测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933841111/job/95132662347


## [Run #31933839449](https://github.com/sgl-project/sglang/actions/runs/31933839449)
- **分支**: `agent/optimize-ideogram-lossless-norm-post`
- **总耗时**: 45.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31933839449

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 44.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933839449/job/95132660211) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业启动后直接进入上传工件阶段，且未找到diffusion-failures文件，说明测试可能提前结束或未执行，但关键错误信息缺失，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933839449/job/95132660211


## [Run #31933836587](https://github.com/sgl-project/sglang/actions/runs/31933836587)
- **分支**: `agent/optimize-cosmos3-t2i-qknorm-rope`
- **总耗时**: 47.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31933836587

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 46.8min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933836587/job/95132646987) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含实际测试执行和失败信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或失败原因被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933836587/job/95132646987


## [Run #31933121612](https://github.com/sgl-project/sglang/actions/runs/31933121612)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 75.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31933121612

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.8min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679276) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 26.9min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679320) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1083秒，未达到性能预期，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679276

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行1378秒后失败，0/4测试通过，属于性能指标未达到预期要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679320

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679156) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679234) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679286) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 74.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679330) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679355) |
| base-b-test-16-npu-a3 / run (0) | 45.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679357) |
| base-b-test-4-npu-a3 / run (1) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679377) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679409) |
| base-b-test-4-npu-a3 / run (0) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679432) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679522) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679526) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143679578) |
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143693468) |
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31933121612/job/95143698296) |


## [Run #31932908145](https://github.com/sgl-project/sglang/actions/runs/31932908145)
- **分支**: `main`
- **总耗时**: 31.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31932908145

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 29.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391042) |
| base-b-test-16-npu-a3 / run (0) | 26.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391055) |
| base-b-test-2-npu-a3 / run (0) | 28.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391060) |
| base-b-test-4-npu-a3 / run (0) | 8.0min | 代码错误 | HiCache MLA测试文件执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391158) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391161) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 28.6min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391197) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95131075067) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 依赖作业失败导致快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95133260343) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常（Accuracy 0.923），但在执行第6个测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391042

- **base-b-test-16-npu-a3 / run (0)**: 测试运行到198/200时，自定义容器实现执行失败，提示联系runner管理员。日志显示测试本身正常（HTTP 200），属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391055

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试运行正常（HTTP 200），但随后出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391060

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试在运行272秒后失败，返回退出码1，导致整个作业失败。具体失败原因需查看该测试文件的详细输出，可能是测试逻辑错误或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391158

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但中途出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391161

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在07:33:38出现错误'Executing the custom container implementation failed'，提示联系runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391197

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试运行正常，但随后出现'Executing the custom container implementation failed'错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95131075067

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业因健康检查发现根因作业base-b-test-4-npu-a3/run失败，触发fast-fail机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95133260343

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391084) |
| base-b-test-8-npu-a3 / run (0) | 10.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391095) |
| base-b-test-4-npu-a3 / run (1) | 27.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391129) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391162) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31932908145/job/95130391243) |


## [Run #31932247281](https://github.com/sgl-project/sglang/actions/runs/31932247281)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 21.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31932247281

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.6min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786656) |
| base-b-test-16-npu-a3 / run (0) | 3.4min | 其他 | 健康检查快速失败，因其他作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786717) |
| base-b-test-1-npu-a3 / run (0) | 20.5min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786724) |
| base-a-test-1-npu-a2 / run (0) | 2.3min | 环境问题 | rustup 下载工具链超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786729) |
| base-b-test-4-npu-a3 / run (0) | 20.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786744) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | 其他 | 作业日志显示所有测试均通过，无失败迹象。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786746) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.8min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786817) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.1min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786838) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.2min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786876) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.2min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786906) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行过程或失败信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或日志被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786656

- **base-b-test-16-npu-a3 / run (0)**: 本作业未实际运行测试，因健康检查检测到同批次中base-a-test-1-npu-a2作业失败，触发fast-fail机制，导致本作业被跳过并报错退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786717

- **base-b-test-1-npu-a3 / run (0)**: 日志显示TokenizerManager和DetokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786724

- **base-a-test-1-npu-a2 / run (0)**: 作业在安装 Rust 1.92 时，从内部缓存服务下载 channel-rust-1.92.toml 超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786729

- **base-b-test-4-npu-a3 / run (0)**: 日志显示torch_npu初始化时出现ImportWarning和RuntimeWarning，随后在Init torch distributed阶段报错"Executing the custom container implementation failed"，表明NPU容器环境配置或驱动存在问题，导致作业无法正常运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786744

- **base-b-test-2-npu-a3 / run (0)**: 日志中6个测试文件全部passed，作业正常完成清理流程，仅有Node 20弃用警告，未发现实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786746

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在07:08:10出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786817

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查结论为failure，触发fast-fail机制，作业未进入实际测试阶段即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786838

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行健康检查时，检测到当前PR的lint检查结论为failure，触发了fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786876

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示服务启动后，在生成请求时出现NPU算子回退警告，随后自定义容器执行失败，可能是NPU设备或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786906

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786715) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31932247281/job/95128786747) |


## [Run #31930934132](https://github.com/sgl-project/sglang/actions/runs/31930934132)
- **分支**: `codex/diffusion-reuse-srt-qwen-vision`
- **总耗时**: 92.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31930934132

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663275) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95127714903) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 39.5min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，未达性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95129217517) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95131669064) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95134833564) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures产物时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663275

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段因健康检查检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际执行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95127714903

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例退出码1，耗时1464秒，可能因性能未达50ms要求或运行错误导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95129217517

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行PR健康检查时，检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95131669064

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业（multimodal-gen-test-1-npu-a3和base-c-test-perf-16-npu-a3），触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95134833564

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663318) |
| base-b-test-4-npu-a3 / run (0) | 26.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663404) |
| base-b-test-16-npu-a3 / run (0) | 51.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663423) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663428) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663435) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663462) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663484) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663534) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663583) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663603) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930934132/job/95125663611) |


## [Run #31930481070](https://github.com/sgl-project/sglang/actions/runs/31930481070)
- **分支**: `main`
- **总耗时**: 60.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31930481070

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 46.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124425984) |
| base-b-test-16-npu-a3 / run (0) | 45.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124425986) |
| multimodal-gen-test-1-npu-a3 | 41.8min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124425999) |
| base-b-test-4-npu-a3 / run (0) | 8.5min | 代码错误 | HiCache MLA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426047) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 46.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426176) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95126061759) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95129197554) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95130100462) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常（HTTP 200），但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124425984

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试在运行到Capturing batches阶段时，自定义容器实现执行失败（Executing the custom container implementation failed），导致作业提前终止，属于自托管runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124425986

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能确认上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124425999

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试在运行271秒后失败，退出码为1，导致整个作业终止。具体失败原因需查看该测试文件的详细输出，可能是测试逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426047

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在运行约46分钟后，日志显示"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境故障，非代码或测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426176

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-4-npu-a3/run(0)，触发快速失败机制，本作业未实际运行即被跳过，属于CI依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95126061759

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行PR测试健康检查时，检测到multimodal-gen-test-1-npu-a3和base-b-test-4-npu-a3两个根因作业已失败，因此本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95129197554

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动前的健康检查中检测到multimodal-gen-test-1-npu-a3和base-b-test-4-npu-a3两个根因作业失败，触发了fast-fail机制，导致本作业被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95130100462

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 11.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124425974) |
| base-b-test-4-npu-a3 / run (1) | 26.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426038) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426059) |
| base-b-test-2-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426091) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426181) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426269) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31930481070/job/95124426330) |


## [Run #31928950250](https://github.com/sgl-project/sglang/actions/runs/31928950250)
- **分支**: `codex/h3-vae-resident-perf`
- **总耗时**: 51.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31928950250

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928950250/job/95120831025) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928950250/job/95120831025


## [Run #31928566924](https://github.com/sgl-project/sglang/actions/runs/31928566924)
- **分支**: `codex/diffusion-reuse-srt-clip`
- **总耗时**: 157.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31928566924

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377891) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95127931824) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 环境问题 | PR健康检查发现其他作业失败，导致本作业被快速跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95129020158) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95135766622) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到最后上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377891

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95127931824

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查检测到multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行测试就被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95129020158

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95135766622

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377887) |
| base-b-test-16-npu-a3 / run (0) | 56.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377910) |
| base-b-test-2-npu-a3 / run (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377921) |
| base-b-test-4-npu-a3 / run (1) | 16.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377950) |
| base-b-test-4-npu-a3 / run (0) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377955) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377957) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120377992) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120378057) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 101.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120378081) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120378088) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95120378124) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928566924/job/95125872664) |


## [Run #31928447033](https://github.com/sgl-project/sglang/actions/runs/31928447033)
- **分支**: `fix/kimi-bcg-multimodal-allowlist`
- **总耗时**: 346.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31928447033

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616348) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | 性能测试未达预期，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95124404776) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616348

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例 test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py 执行失败，耗时1109秒，未通过性能测试，可能因模型性能未达阈值或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95124404776

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 13.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616259) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616266) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616268) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616274) |
| base-b-test-4-npu-a3 / run (0) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616276) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616287) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616375) |
| base-b-test-16-npu-a3 / run (0) | 52.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616376) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 34.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616415) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 114.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616442) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95119616495) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 272.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95126425075) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95127445222) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928447033/job/95135834162) |


## [Run #31928029285](https://github.com/sgl-project/sglang/actions/runs/31928029285)
- **分支**: `sync/c6998c515-model-input-embed-width`
- **总耗时**: 74.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31928029285

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 38.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594478) |
| base-b-test-16-npu-a3 / run (0) | 16.3min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594504) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594516) |
| base-b-test-8-npu-a3 / run (0) | 1.2min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594537) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594555) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594569) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594591) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594603) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594604) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594622) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594708) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（未找到文件）及清理步骤，未出现任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594478

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594504

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，触发fast-fail，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594516

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594537

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594555

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，发现同一运行中另一个作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594569

- **base-b-test-4-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤后触发fast-fail机制，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594591

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594603

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594604

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示根因作业为multimodal-gen-test-1-npu-a3，本作业因健康检查过滤级联失败而被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594622

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594708

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31928029285/job/95118594493) |


## [Run #31927954354](https://github.com/sgl-project/sglang/actions/runs/31927954354)
- **分支**: `sync/637c210863-chat-header-overrides`
- **总耗时**: 59.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927954354

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436763) |
| base-b-test-2-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436821) |
| base-b-test-16-npu-a3 / run (0) | 1.2min | 其他 | 健康检查触发快速失败，根因作业为multimodal-gen-test-1-npu-a3 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436838) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436855) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436861) |
| base-b-test-4-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436865) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436884) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436946) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436999) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118437008) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118437039) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅显示Node.js 20弃用警告和diffusion-failures目录无文件上传提示，未包含测试执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436763

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被终止，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436821

- **base-b-test-16-npu-a3 / run (0)**: 该作业因其他作业（multimodal-gen-test-1-npu-a3）健康检查失败而被级联跳过，并非自身代码或环境问题，属于CI流程中的级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436838

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436855

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436861

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436865

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436884

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段因健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业被跳过未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436946

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436999

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查过滤后根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118437008

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118437039

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927954354/job/95118436813) |


## [Run #31927917567](https://github.com/sgl-project/sglang/actions/runs/31927917567)
- **分支**: `oss/l2-transfer-consolidation`
- **总耗时**: 155.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927917567

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270702) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95123197016) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 47.0min | 性能回归 | 性能测试用例 qwen3_235b_w8a8_8p_in3k5_out1k5_50ms 失败，未达到性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95125815324) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 23.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95131041707) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到具体测试命令和错误输出。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败产物或失败原因未记录。需查看完整日志才能定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270702

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1118秒后失败，0/1通过，属于性能测试未达标，可能因模型推理速度低于50ms目标或环境波动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95123197016

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 在 NPU 性能测试中，qwen3_235b_a22b 用例执行失败（exit code 1），而其他两个用例通过。该用例耗时 1483 秒，可能因吞吐量或延迟未达 50ms 性能指标而判定失败，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95125815324

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示性能测试正常运行中，但突然报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95131041707

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270746) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270760) |
| base-b-test-2-npu-a3 / run (0) | 19.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270768) |
| base-b-test-16-npu-a3 / run (0) | 54.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270785) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270786) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270801) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270837) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270890) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270901) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270912) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95118270920) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927917567/job/95126368937) |


## [Run #31927906274](https://github.com/sgl-project/sglang/actions/runs/31927906274)
- **分支**: `sync/52d1a85cf-unified-swa-page-map`
- **总耗时**: 135.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927906274

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.8min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95123491672) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95125549159) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因同PR中另一个作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95126516432) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 其他 | 该作业因其他根因作业失败被快速失败跳过，自身未执行测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95131348040) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1070秒后失败，退出码1，0/1测试通过，属于性能指标未达标或执行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95123491672

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95125549159

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示health-check检测到同PR的base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95126516432

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业被级联跳过，日志中无实际测试执行或失败信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95131348040

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292598) |
| base-b-test-1-npu-a3 / run (0) | 25.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292622) |
| base-b-test-4-npu-a3 / run (1) | 13.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292635) |
| base-b-test-16-npu-a3 / run (0) | 56.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292646) |
| base-b-test-4-npu-a3 / run (0) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292657) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292678) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292731) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292802) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292832) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292908) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927906274/job/95118292965) |


## [Run #31927889918](https://github.com/sgl-project/sglang/actions/runs/31927889918)
- **分支**: `flashinfer-sm90-mxfp4-fp8`
- **总耗时**: 49.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927889918

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.3min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212414) |
| base-b-test-4-npu-a3 / run (0) | 0.9min | 环境问题 | 健康检查发现其他作业失败导致本作业被快速跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212417) |
| base-b-test-16-npu-a3 / run (0) | 1.4min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212436) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212451) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212460) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212482) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查失败，根因是多模态生成测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212561) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查失败，根因是多模态测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212571) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.1min | 其他 | 健康检查快速失败，根因是其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212574) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查失败，根因是多模态生成测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212578) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212675) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures目录，但未找到文件，未展示实际测试命令或失败原因，可能因日志截断或作业提前结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212414

- **base-b-test-4-npu-a3 / run (0)**: health-check检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212417

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤，最终因快速失败策略终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212436

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail跳过本作业，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212451

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212460

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212482

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212561

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212571

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212574

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212578

- **base-b-test-4-npu-a3 / run (1)**: 该作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212675

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927889918/job/95118212511) |


## [Run #31927745761](https://github.com/sgl-project/sglang/actions/runs/31927745761)
- **分支**: `sync/b1b1f1038-eplb-report-modes`
- **总耗时**: 50.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927745761

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916440) |
| multimodal-gen-test-1-npu-a3 | 36.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916454) |
| base-b-test-16-npu-a3 / run (0) | 2.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916463) |
| base-b-test-2-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916471) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916480) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916495) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916502) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现根因作业失败，导致级联跳过本作业 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916589) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916602) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败被快速失败跳过，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916605) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.0min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916613) |

- **base-b-test-4-npu-a3 / run (0)**: 健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤，最终因快速失败策略终止，非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916440

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916454

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败被终止，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916463

- **base-b-test-2-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，属于根因失败，因此本作业（base-b-test-2-npu-a3）被快速失败跳过，并非自身执行出错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916471

- **base-b-test-8-npu-a3 / run (0)**: 该作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916480

- **base-b-test-4-npu-a3 / run (1)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，导致本作业未实际运行测试即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916495

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因作业触发fast-fail，导致本作业（base-b-test-1-npu-a3）未实际运行即被终止，属于依赖的上游失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916502

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示本作业因PR健康检查失败被跳过，根因是multimodal-gen-test-1-npu-a3作业失败，其他多个作业被过滤为级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916589

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因Fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916602

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因失败为multimodal-gen-test-1-npu-a3，本作业因fast-fail机制被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916605

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916613

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927745761/job/95117916549) |


## [Run #31927569122](https://github.com/sgl-project/sglang/actions/runs/31927569122)
- **分支**: `fix/hicache-component-scoped-load-back`
- **总耗时**: 137.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927569122

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 43.9min | 其他 | 作业正常结束，无失败迹象，仅上传artifact时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417267) |
| base-b-test-4-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417510) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 28.9min | 环境问题 | Kubernetes Pod 启动失败，状态为 Failed，导致作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417597) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 2.0min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95120399367) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95123926260) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.0min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95130650368) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业成功完成，upload-artifact步骤提示未找到diffusion-failures/目录下的文件，因此未上传任何artifact，但这不是失败原因，作业整体正常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417267

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被过滤后，因根因作业失败而触发fast-fail机制，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417510

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 Pod linux-aarch64-a3-16-cn12-001-772vk-runner-xjmnz-workflow 不健康且状态为 Failed，可能是资源不足、镜像拉取失败或节点问题，属于基础设施环境故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417597

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95120399367

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3两个根因作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95123926260

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-acc-16-npu-a3为根因失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95130650368

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417342) |
| base-a-test-1-npu-a2 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417388) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417390) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417397) |
| base-b-test-16-npu-a3 / run (0) | 45.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417404) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417448) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417526) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417579) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 103.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927569122/job/95117417630) |


## [Run #31927554425](https://github.com/sgl-project/sglang/actions/runs/31927554425)
- **分支**: `sync/778b30ce0-world-size-one-aliasing`
- **总耗时**: 67.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927554425

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因任务失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434727) |
| multimodal-gen-test-1-npu-a3 | 36.6min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434731) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434742) |
| base-b-test-4-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434750) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434759) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434762) |
| base-b-test-8-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434793) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434853) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.7min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434899) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434905) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434922) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434727

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions的初始化、Node版本警告和上传artifact步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434731

- **base-b-test-4-npu-a3 / run (1)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434742

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434750

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被终止，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434759

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434762

- **base-b-test-8-npu-a3 / run (0)**: 该作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，因此主动跳过执行并报错，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434793

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434853

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434899

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434905

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434922

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927554425/job/95117434821) |


## [Run #31927523236](https://github.com/sgl-project/sglang/actions/runs/31927523236)
- **分支**: `sync/c96e2b686-post-capture-reserve`
- **总耗时**: 126.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927523236

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347237) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95122298094) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95123599472) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95128866784) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（未找到文件），未显示任何测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347237

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95122298094

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95123599472

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95128866784

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347267) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347274) |
| base-b-test-8-npu-a3 / run (0) | 12.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347279) |
| base-b-test-4-npu-a3 / run (1) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347289) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347300) |
| base-b-test-16-npu-a3 / run (0) | 53.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347328) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347377) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347390) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347441) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347487) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95117347549) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927523236/job/95120228363) |


## [Run #31927485489](https://github.com/sgl-project/sglang/actions/runs/31927485489)
- **分支**: `sync/efb38a842-mm-placeholder-count`
- **总耗时**: 43.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927485489

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261882) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261886) |
| base-a-test-1-npu-a2 / run (0) | 4.6min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261898) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261904) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现多个NPU测试作业级联失败，根因是其他作业失败导致快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261905) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 作业因健康检查快速失败机制被跳过，非自身测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261926) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261969) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262000) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262069) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262079) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.8min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262080) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 作业因健康检查快速失败机制被跳过，非自身测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262086) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或日志被截断，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261882

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动前的健康检查中检测到两个根因失败作业（base-a-test-1-npu-a2和base-c-test-acc-8-npu-a3），根据策略触发fast-fail，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261886

- **base-a-test-1-npu-a2 / run (0)**: test/registered/unit/managers/test_mm_embedding_length.py 缺少 `if __name__ == "__main__":` 入口，导致pytest风格测试被静默跳过，collect_tests抛出ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261898

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到base-a-test-1-npu-a2和base-c-test-acc-8-npu-a3两个根因作业失败，本作业因快速失败（fast-fail）被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261904

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因作业为base-a-test-1-npu-a2和base-c-test-acc-8-npu-a3，当前作业因快速失败机制被跳过，非自身代码问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261905

- **base-b-test-2-npu-a3 / run (0)**: 该作业被健康检查识别为级联失败（failed step: Check PR test health），根因是其他作业（base-a-test-1-npu-a2 和 base-c-test-acc-8-npu-a3）失败，导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261926

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为base-a-test-1-npu-a2和base-c-test-acc-8-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117261969

- **base-b-test-4-npu-a3 / run (1)**: 日志显示本作业因健康检查检测到根因作业（base-a-test-1-npu-a2和base-c-test-acc-8-npu-a3）失败而被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262000

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业base-a-test-1-npu-a2/run (0)，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262069

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因失败作业（base-a-test-1-npu-a2 和 base-c-test-acc-8-npu-a3），本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262079

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: test/registered/unit/managers/test_mm_embedding_length.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 python3 file.py -f 下静默跳过，collect_tests 抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262080

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示该作业在启动前被健康检查过滤，根因是其他作业（base-a-test-1-npu-a2和base-c-test-acc-8-npu-a3）失败触发了fast-fail，导致本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927485489/job/95117262086


## [Run #31927191790](https://github.com/sgl-project/sglang/actions/runs/31927191790)
- **分支**: `mmangkad/restore-router-gemm-threshold`
- **总耗时**: 113.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927191790

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519289) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.8min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95119259619) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95120927285) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | PR健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95123238101) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95127401422) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件）等常规信息，未出现测试执行或失败的具体错误，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519289

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1155秒，未通过性能基准要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95119259619

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业（multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3），触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95120927285

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因失败作业，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95123238101

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95127401422

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519277) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519304) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519311) |
| base-b-test-16-npu-a3 / run (0) | 52.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519320) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519325) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519326) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519363) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519409) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519421) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519422) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927191790/job/95116519506) |


## [Run #31927156822](https://github.com/sgl-project/sglang/actions/runs/31927156822)
- **分支**: `mmangkad/fix-gptq-scheme-attach`
- **总耗时**: 147.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31927156822

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 44.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465221) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 131.2min | 精度回归 | qwen3_5_9b GSM8K 精度测试失败，导致作业整体退出码非零。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465501) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.7min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95118403211) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 45.5min | 性能回归 | 性能测试中qwen3_235b用例失败，疑似性能不达标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95120274960) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业整体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465221

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k 用例失败（exit code 1），其余两个用例通过。该用例属于精度测试，失败表明模型输出精度未达预期，可能由代码改动或环境差异引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465501

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1141秒后退出码1，0/1通过，属于性能指标未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95118403211

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 在base-c-test-perf-16-npu-a3作业中，qwen3_235b_a22b的w8a8性能测试退出码为1，而其他两个用例通过，表明该用例存在性能回归或未达阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95120274960

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 57.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465207) |
| base-b-test-8-npu-a3 / run (0) | 8.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465236) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465240) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465295) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465302) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465308) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465336) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465463) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465470) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95116465504) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31927156822/job/95122335966) |


## [Run #31926709432](https://github.com/sgl-project/sglang/actions/runs/31926709432)
- **分支**: `main`
- **总耗时**: 95.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31926709432

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 45.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传工件信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379275) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379317) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 76.1min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379383) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.5min | 性能回归 | 性能测试用例未通过，可能是性能未达标或执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95117074192) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.2min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95119828060) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95120886805) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅显示Node.js 20弃用警告、上传diffusion-failures工件时未找到文件等提示，未包含测试执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379275

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试执行失败，退出码为1，耗时281秒，导致整个作业失败。具体失败原因需查看该测试文件的详细输出，可能涉及功能实现或测试断言问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379317

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在06:03:22时出现'Executing the custom container implementation failed'错误，提示联系runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379383

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试 test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py 失败，退出码1，耗时1187秒，未达到预期性能指标，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95117074192

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95119828060

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现multimodal-gen-test-1-npu-a3、base-b-test-4-npu-a3等根因作业失败，本作业作为级联失败被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95120886805

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 46.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379253) |
| base-b-test-8-npu-a3 / run (0) | 10.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379255) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379257) |
| base-b-test-16-npu-a3 / run (0) | 70.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379293) |
| base-b-test-2-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379295) |
| base-b-test-4-npu-a3 / run (1) | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379315) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379372) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379373) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926709432/job/95115379414) |


## [Run #31926577809](https://github.com/sgl-project/sglang/actions/runs/31926577809)
- **分支**: `codex/h3-vae-resident-perf`
- **总耗时**: 41.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31926577809

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926577809/job/95115033495) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，只有GitHub Actions的常规准备、上传工件（未找到文件）和清理步骤。无法判断失败原因，可能是测试未运行或日志被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926577809/job/95115033495


## [Run #31926286150](https://github.com/sgl-project/sglang/actions/runs/31926286150)
- **分支**: `main`
- **总耗时**: 6.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31926286150

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377739) |
| base-b-test-2-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377740) |
| multimodal-gen-test-1-npu-a3 | 5.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377746) |
| base-b-test-4-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377779) |
| base-b-test-4-npu-a3 / run (1) | 1.0min | 环境问题 | 自定义容器启动失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377794) |
| base-b-test-8-npu-a3 / run (0) | 1.2min | 环境问题 | 自定义容器启动失败，导致作业在初始化阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377798) |
| base-b-test-16-npu-a3 / run (0) | 1.3min | 环境问题 | 自定义容器执行失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377820) |
| base-b-test-1-npu-a3 / run (0) | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377834) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377959) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377974) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377981) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377984) |

- **base-a-test-1-npu-a2 / run (0)**: 第二个测试test_npu_ascend_dsv4_backend.py启动后，自定义容器实现执行失败，导致作业终止。第一个测试已通过，问题出在容器环境而非测试代码本身。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377739

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377740

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未出现测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377746

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377779

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在安装依赖包时，执行自定义容器实现失败，提示联系自托管 runner 管理员，属于 runner 环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377794

- **base-b-test-8-npu-a3 / run (0)**: 日志显示在安装依赖后，执行自定义容器实现时出错，错误信息为“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于环境配置或容器兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377798

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在安装依赖包时，GitHub Actions 报错“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于 runner 或容器环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377820

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377834

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377959

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377974

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377981

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926286150/job/95114377984


## [Run #31926257258](https://github.com/sgl-project/sglang/actions/runs/31926257258)
- **分支**: `codex/pi05-reuse-srt-siglip`
- **总耗时**: 85.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31926257258

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 40.5min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259310) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.5min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114729690) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业被健康检查快速失败机制跳过，因同一次运行中另一个作业（base-c-test-perf-8-npu-a3）已失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95116689227) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95118188481) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 该作业因其他根因作业失败被快速跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95122229716) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions运行环境准备、Node版本警告及上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259310

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行约1070秒后退出码为1，属于性能测试未通过，可能因吞吐或延迟未达预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114729690

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到根因失败作业为base-c-test-perf-8-npu-a3，触发fast-fail，导致本作业未实际执行测试即被跳过，属于依赖作业失败的连锁反应。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95116689227

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3，本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95118188481

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3，本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95122229716

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259209) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259216) |
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259244) |
| base-b-test-2-npu-a3 / run (0) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259286) |
| base-b-test-16-npu-a3 / run (0) | 63.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259300) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259314) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259330) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259519) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259537) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259543) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926257258/job/95114259908) |


## [Run #31926254363](https://github.com/sgl-project/sglang/actions/runs/31926254363)
- **分支**: `fix/prefill-input-embeds-width`
- **总耗时**: 87.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31926254363

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 44.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114274532) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95116142773) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95118299622) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.4min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95119373488) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95122466453) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅显示Node.js版本弃用警告和上传artifacts时无文件。可能因日志截断或作业在测试前已结束，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114274532

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1142秒后退出码为1，0/1测试通过，属于性能指标未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95116142773

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段因健康检查检测到同批次中multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95118299622

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3，本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95119373488

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业在启动阶段因其他根因作业（multimodal-gen-test-1-npu-a3、base-c-test-perf-8-npu-a3）失败而被级联跳过，自身未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95122466453

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114273834) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114273837) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114273877) |
| base-b-test-16-npu-a3 / run (0) | 52.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114273924) |
| base-b-test-4-npu-a3 / run (0) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114273939) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114273943) |
| base-b-test-8-npu-a3 / run (0) | 9.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114273951) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114274752) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114274761) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114275351) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926254363/job/95114275488) |


## [Run #31926215311](https://github.com/sgl-project/sglang/actions/runs/31926215311)
- **分支**: `codex/srt-vision-sdpa-reshape`
- **总耗时**: 95.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31926215311

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 59.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146666) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.1min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114669354) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.4min | 其他 | 健康检查发现同PR中另一个作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95116620724) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95118027495) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 作业因其他根因作业失败而被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95123103836) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体信息，仅有Node.js弃用警告和上传artifact时未找到文件的提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146666

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1132秒后失败，该测试为性能测试，预计耗时3600秒，但未通过性能指标要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114669354

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95116620724

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而快速失败，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95118027495

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3，本作业被级联跳过，属于CI快速失败机制，非本作业自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95123103836

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146738) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146811) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146841) |
| base-b-test-16-npu-a3 / run (0) | 52.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146845) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146892) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146921) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114146958) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114147013) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 93.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114147052) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114147071) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926215311/job/95114147110) |


## [Run #31926035114](https://github.com/sgl-project/sglang/actions/runs/31926035114)
- **分支**: `fix/prefill-input-embeds-width`
- **总耗时**: 5.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31926035114

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志被截断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720488) |
| base-b-test-4-npu-a3 / run (1) | 1.6min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720540) |
| base-b-test-1-npu-a3 / run (0) | 2.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720580) |
| base-a-test-1-npu-a2 / run (0) | 4.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720584) |
| base-b-test-4-npu-a3 / run (0) | 1.4min | 环境问题 | 自定义容器启动失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720630) |
| base-b-test-8-npu-a3 / run (0) | 3.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720636) |
| base-b-test-2-npu-a3 / run (0) | 2.4min | 环境问题 | 自定义容器执行失败，镜像拉取中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720645) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.2min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720885) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.9min | 环境问题 | 自定义容器执行失败，torch-npu安装过程中出错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720890) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.8min | 环境问题 | 容器执行失败，下载triton-ascend依赖时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720902) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传artifact步骤，提示未找到diffusion-failures目录，无实际测试输出或错误信息，可能因前置步骤失败或日志不完整导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720488

- **base-b-test-4-npu-a3 / run (1)**: 在安装torch-npu依赖时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720540

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在安装sglang_router包后，执行自定义容器实现时失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720580

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行测试前，执行自定义容器实现时失败，报错提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720584

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在安装系统包时出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720630

- **base-b-test-8-npu-a3 / run (0)**: 作业在运行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720636

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在下载容器镜像过程中（约72%进度）出现错误，提示“Executing the custom container implementation failed”，属于自托管runner环境或镜像拉取问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720645

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在准备阶段执行自定义容器实现时失败，错误为"Executing the custom container implementation failed"，属于runner环境或容器配置问题，并非测试代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720885

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 在安装torch-npu==2.10.0时，容器执行失败，错误为"Executing the custom container implementation failed"，可能是容器环境或依赖问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720890

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在安装triton-ascend==3.2.1.dev20260530时，自定义容器实现执行失败，可能是网络或镜像问题导致依赖下载中断，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720902

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31926035114/job/95113720831) |


## [Run #31925695428](https://github.com/sgl-project/sglang/actions/runs/31925695428)
- **分支**: `main`
- **总耗时**: 15.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31925695428

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.8min | 其他 | 日志不完整，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874098) |
| base-b-test-4-npu-a3 / run (0) | 8.8min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874099) |
| base-b-test-4-npu-a3 / run (1) | 12.4min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874107) |
| base-b-test-16-npu-a3 / run (0) | 11.7min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874110) |
| base-b-test-1-npu-a3 / run (0) | 11.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874114) |
| base-b-test-2-npu-a3 / run (0) | 12.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874118) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874285) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874290) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.1min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874307) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 9.2min | 超时 | TokenizerManager watchdog超时导致容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95113330131) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行后上传diffusion-failures目录时提示无文件，未展示测试执行细节或错误信息，无法判断具体失败原因，可能为测试未产生失败样本或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874098

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试执行失败，退出码为1，耗时281秒。日志未显示具体错误原因，但测试未通过，属于测试代码或被测功能存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874099

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中容器突然报错“Executing the custom container implementation failed”，随后作业被终止，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874107

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动NPU容器后，TokenizerManager初始化过程中自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874110

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行过程中出现“Executing the custom container implementation failed”错误，可能是容器环境或资源问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874114

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试用例已通过（输出'.'），但随后出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874118

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但执行自定义容器时失败，错误为'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874285

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在加载模型分片时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874290

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装依赖过程中，自定义容器实现执行失败（Executing the custom container implementation failed），提示联系自托管runner管理员，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874307

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示TokenizerManager watchdog超时（300秒），服务进程卡死，最终导致自定义容器执行失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95113330131

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874137) |
| base-b-test-8-npu-a3 / run (0) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874181) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925695428/job/95112874339) |


## [Run #31925388557](https://github.com/sgl-project/sglang/actions/runs/31925388557)
- **分支**: `fix/bcg-deepstack-replay-slot`
- **总耗时**: 71.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31925388557

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 35.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079578) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 68.1min | 精度回归 | NPU精度测试中qwen3_5_9b_bf16_1p_gsm8k用例失败，0/3测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079727) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.4min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112963991) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95116118905) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95116210621) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业整体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079578

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件base-c-test-acc-2-npu-a3中，qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，耗时3851秒，所有3个测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079727

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1069秒后退出码为1，属于性能测试未通过，可能因吞吐或延迟未达阈值导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112963991

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业未实际运行，因同次运行中其他作业（multimodal-gen-test-1-npu-a3 和 base-c-test-perf-8-npu-a3）失败，触发 fast-fail 跳过，属于上游失败导致的连带跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95116118905

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95116210621

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079576) |
| base-b-test-16-npu-a3 / run (0) | 46.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079589) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079603) |
| base-b-test-4-npu-a3 / run (0) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079629) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079651) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079673) |
| base-b-test-2-npu-a3 / run (0) | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079693) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079725) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079745) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925388557/job/95112079758) |


## [Run #31925270179](https://github.com/sgl-project/sglang/actions/runs/31925270179)
- **分支**: `main`
- **总耗时**: 11.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31925270179

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 7.3min | 环境问题 | 自定义容器执行失败，测试在完成一个请求后异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806243) |
| base-b-test-2-npu-a3 / run (0) | 5.5min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806257) |
| base-b-test-4-npu-a3 / run (1) | 6.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806258) |
| multimodal-gen-test-1-npu-a3 | 9.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806283) |
| base-b-test-1-npu-a3 / run (0) | 3.5min | 环境问题 | 自定义容器执行失败，导致作业在测试启动前中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806297) |
| base-b-test-4-npu-a3 / run (0) | 5.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806322) |
| base-b-test-16-npu-a3 / run (0) | 7.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806339) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.7min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806369) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.4min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806397) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.7min | 环境问题 | 自定义容器执行失败，导致作业在安装依赖阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806436) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 3.6min | 环境问题 | 自定义容器执行失败，导致测试未开始即终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95112392592) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试已成功完成一个200请求的推理任务，但随后出现“Executing the custom container implementation failed”错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806243

- **base-b-test-2-npu-a3 / run (0)**: 日志显示服务启动后立即报错"Executing the custom container implementation failed"，且检测到SymmetricMemory不支持cuda设备类型，NPU后端算子回退CPU，表明容器环境配置或NPU驱动存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806257

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，提示联系runner管理员，属于NPU CI环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806258

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806283

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试文件已列出但未实际运行，随后报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806297

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试在捕获批次时（bs=176）突然报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806322

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型权重加载成功，但随后报错“Executing the custom container implementation failed”，表明自托管runner的容器环境在执行过程中崩溃或配置错误，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806339

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在加载模型分片时（约60%进度），自定义容器实现执行失败，提示联系自托管runner管理员。可能是NPU设备、容器或资源问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806369

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装依赖后，执行自定义容器时出现错误："Executing the custom container implementation failed"，可能是容器环境或配置问题，导致作业无法继续。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806397

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在构建jieba等wheel包时，出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806436

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动测试命令后立即报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，与测试代码无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95112392592

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806264) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925270179/job/95111806374) |


## [Run #31925179201](https://github.com/sgl-project/sglang/actions/runs/31925179201)
- **分支**: `codex/k3-mm-cache-k3`
- **总耗时**: 128.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31925179201

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592332) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95112205175) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 30.9min | 性能回归 | NPU性能测试用例执行失败，0/4通过，首个失败用例为qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95114266742) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95115474374) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.2min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95123746725) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含multimodal-gen-test的具体执行过程或失败信息，仅显示上传diffusion-failures工件时未找到文件，可能测试未运行或已通过，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592332

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行约1140秒后失败，该测试为性能测试，预期耗时3600秒，但实际未通过，可能因性能指标未达到要求或执行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95112205175

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中qwen3_235b_a22b模型用例返回退出码1，耗时1472秒接近预估3600秒上限，可能因性能未达标或执行异常导致失败，需检查具体性能指标是否满足阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95114266742

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95115474374

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在健康检查阶段因其他作业（multimodal-gen-test-1-npu-a3、base-c-test-perf-8/16-npu-a3）失败而被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95123746725

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592365) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592384) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592396) |
| base-a-test-1-npu-a2 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592409) |
| base-b-test-4-npu-a3 / run (0) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592434) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592457) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 125.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592463) |
| base-b-test-16-npu-a3 / run (0) | 51.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592464) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592468) |
| base-b-test-2-npu-a3 / run (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592480) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31925179201/job/95111592481) |


## [Run #31924636061](https://github.com/sgl-project/sglang/actions/runs/31924636061)
- **分支**: `codex/h3-vae-resident-perf`
- **总耗时**: 7.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31924636061

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31924636061/job/95110179187) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node.js版本弃用警告和上传artifact时无文件可传的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31924636061/job/95110179187


## [Run #31924313612](https://github.com/sgl-project/sglang/actions/runs/31924313612)
- **分支**: `codex/diffusion-biweekly-review-20260813`
- **总耗时**: 40.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31924313612

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31924313612/job/95109384042) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions运行环境准备、Node版本警告及上传artifact时未找到diffusion-failures目录的提示，无法判断测试失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31924313612/job/95109384042


## [Run #31924180496](https://github.com/sgl-project/sglang/actions/runs/31924180496)
- **分支**: `main`
- **总耗时**: 27.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31924180496

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.4min | 环境问题 | 作业因缺少失败产物文件而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31924180496/job/95109101087) |

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures目录时提示无文件，说明测试未产生失败样本，作业可能因环境配置或前置步骤异常而中断，未进入核心测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31924180496/job/95109101087


## [Run #31922969081](https://github.com/sgl-project/sglang/actions/runs/31922969081)
- **分支**: `main`
- **总耗时**: 31.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31922969081

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 23.3min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870726) |
| base-b-test-4-npu-a3 / run (1) | 27.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870738) |
| base-b-test-8-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870747) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870749) |
| base-b-test-2-npu-a3 / run (0) | 24.5min | 环境问题 | 自定义容器执行失败，NPU测试在加载权重后崩溃。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870768) |
| base-b-test-16-npu-a3 / run (0) | 20.6min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870778) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 24.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870836) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 27.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870843) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870884) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870885) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载完成后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870726

- **base-b-test-4-npu-a3 / run (1)**: 日志显示服务器已成功启动并完成测试请求，但随后出现"Executing the custom container implementation failed"错误，表明自托管runner的容器实现存在问题，而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870738

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到根因作业 base-b-test-4-npu-a3 / run (0) 已失败，因此触发 fast-fail 机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870747

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 执行失败（exit code 1），耗时281秒，导致整个作业失败。具体失败原因需查看该测试的详细输出，可能涉及功能实现或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870749

- **base-b-test-2-npu-a3 / run (0)**: 日志显示模型权重加载成功，但随后出现"Executing the custom container implementation failed"错误，提示联系self-hosted runner管理员，属于NPU容器环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870768

- **base-b-test-16-npu-a3 / run (0)**: 日志显示TokenizerManager watchdog超时（300秒），模型分片加载耗时过长（161个分片加载4分21秒），最终自定义容器执行失败，属于NPU环境性能或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870778

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870836

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870843

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试运行正常，但在03:24:52出现错误“Executing the custom container implementation failed”，提示联系self-hosted runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870884

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业 base-b-test-4-npu-a3 / run (0) 失败，触发了 fast-fail 机制，本作业未实际执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870885

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922969081/job/95105870736) |


## [Run #31922959336](https://github.com/sgl-project/sglang/actions/runs/31922959336)
- **分支**: `fix-mxfp4-sharded-state`
- **总耗时**: 90.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31922959336

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.2min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95106565640) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因健康检查发现其他根因作业失败而被快速跳过，并非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95108987038) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95110007260) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败而被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95114638112) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1152秒。该测试为性能测试，可能因推理延迟或吞吐量未达到设定阈值（如50ms）而判定失败，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95106565640

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，导致本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95108987038

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动前的健康检查中检测到 base-c-test-perf-8-npu-a3 作业失败，触发 fast-fail 机制，本作业未实际运行即被取消，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95110007260

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（2-npu）被级联过滤并快速失败，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95114638112

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105858810) |
| base-b-test-16-npu-a3 / run (0) | 55.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105858821) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105858826) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105858834) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105858836) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105858858) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105858859) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105858864) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105859054) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105859062) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105859075) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922959336/job/95105859153) |


## [Run #31922891724](https://github.com/sgl-project/sglang/actions/runs/31922891724)
- **分支**: `RM/glm52-fuse-hadamard-quant`
- **总耗时**: 86.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31922891724

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试用例minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms失败，未达到性能要求。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95106066330) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 28.8min | 性能回归 | NPU性能测试未达标，4个测试全部失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95108080490) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95114253526) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1098秒。该用例为性能测试，可能因推理延迟或吞吐量未达标导致失败，需检查具体性能指标是否满足阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95106066330

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行1480秒后退出码1，所有性能用例均未通过，疑似性能指标未达预期或运行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95108080490

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3和16-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95114253526

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614451) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614515) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614537) |
| base-b-test-8-npu-a3 / run (0) | 8.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614540) |
| base-b-test-4-npu-a3 / run (0) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614568) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614592) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614650) |
| base-b-test-16-npu-a3 / run (0) | 45.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614674) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614723) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614756) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614759) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95105614795) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922891724/job/95109674629) |


## [Run #31922519115](https://github.com/sgl-project/sglang/actions/runs/31922519115)
- **分支**: `feat/triton-sparse-mla`
- **总耗时**: 126.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31922519115

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 28.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635256) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.0min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8测试返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95105017243) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95107111003) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95108837071) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95116878154) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业整体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635256

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1117秒后失败，0/1通过，可能因性能未达预期或运行错误导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95105017243

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查结论为failure，触发fast-fail机制，作业未进入实际测试即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95107111003

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查结论为failure，触发fast-fail机制直接终止，未进入实际性能测试。属于PR代码风格或静态检查问题，非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95108837071

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 作业在启动阶段执行PR健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95116878154

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 59.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635283) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635300) |
| base-b-test-4-npu-a3 / run (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635307) |
| base-b-test-2-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635339) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635373) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635408) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635458) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 121.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635462) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635490) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635493) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922519115/job/95104635523) |


## [Run #31922197965](https://github.com/sgl-project/sglang/actions/runs/31922197965)
- **分支**: `RM/glm52-ptpc-fp8-proj`
- **总耗时**: 88.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31922197965

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.4min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95104287370) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 44.0min | 性能回归 | qwen3_235b_a22b性能测试未通过，退出码1，可能未达性能阈值。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95106316754) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95107966652) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8卡、16卡）已失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95112651577) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1077秒后失败，退出码1，0/1通过。该测试为性能测试，可能因推理延迟或吞吐量未达阈值导致失败，需检查性能指标是否回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95104287370

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例失败，耗时1408秒，其他两个用例通过，疑似该模型性能未达标或存在回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95106316754

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，根因失败作业被过滤后触发 fast-fail，导致本作业未实际运行即被终止，属于上游作业失败引发的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95107966652

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业未实际运行，因PR健康检查检测到base-c-test-perf-8-npu-a3和16-npu-a3作业失败，触发fast-fail机制，跳过本作业执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95112651577

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827077) |
| multimodal-gen-test-1-npu-a3 | 34.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827086) |
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827103) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827110) |
| base-b-test-16-npu-a3 / run (0) | 51.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827136) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827152) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827156) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827159) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827250) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827280) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827289) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922197965/job/95103827330) |


## [Run #31922040941](https://github.com/sgl-project/sglang/actions/runs/31922040941)
- **分支**: `main`
- **总耗时**: 24.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31922040941

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 23.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434861) |
| base-b-test-16-npu-a3 / run (0) | 23.0min | 环境问题 | NPU容器执行失败，自定义容器实现报错 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434915) |
| base-b-test-2-npu-a3 / run (0) | 23.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434924) |
| base-b-test-4-npu-a3 / run (1) | 22.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434951) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434980) |
| base-b-test-4-npu-a3 / run (0) | 8.3min | 超时 | NPU测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435034) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 22.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435158) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 22.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435178) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 19.0min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103883981) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 0.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95105744989) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434861

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载MoE模型权重时出现PyTorch copy_操作异常，随后Scheduler watchdog超时，最终自定义容器执行失败，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434915

- **base-b-test-2-npu-a3 / run (0)**: 日志显示服务已成功启动，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434924

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中（进度约60%）时，runner报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434951

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于NPU测试环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434980

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test_npu_hicache_mla.py运行301秒后超时（预计400秒），返回退出码1，导致整个作业失败。日志显示测试未通过，且无具体错误信息，属于执行超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435034

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在02:54:21出现错误'Executing the custom container implementation failed'，提示联系self hosted runner管理员，属于runner环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435158

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435178

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在性能测试运行过程中，runner报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能回归导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103883981

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95105744989

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103434904) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435007) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435133) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31922040941/job/95103435210) |


## [Run #31921994026](https://github.com/sgl-project/sglang/actions/runs/31921994026)
- **分支**: `codex/component-residency-policy`
- **总耗时**: 51.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31921994026

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 50.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921994026/job/95103261843) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921994026/job/95103261843


## [Run #31921817121](https://github.com/sgl-project/sglang/actions/runs/31921817121)
- **分支**: `codex/k3-mm-cache-k3`
- **总耗时**: 86.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31921817121

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767649) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95103248246) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.8min | 其他 | 健康检查发现同PR中另一个作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95105332834) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95107361690) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在03:50:01出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767649

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1117秒后失败，0/1通过，属于性能测试未满足阈值要求，可能因模型推理延迟或吞吐量未达标。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95103248246

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，将其标记为根因作业，随后本作业因fast-fail策略被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95105332834

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 本作业在启动前的PR健康检查阶段，检测到同PR中base-c-test-perf-8-npu-a3作业已失败，触发fast-fail机制，本作业被跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95107361690

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 55.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767267) |
| multimodal-gen-test-1-npu-a3 | 39.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767297) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767307) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767328) |
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767332) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767340) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767349) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767456) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767600) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767618) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921817121/job/95102767639) |


## [Run #31921154636](https://github.com/sgl-project/sglang/actions/runs/31921154636)
- **分支**: `main`
- **总耗时**: 23.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31921154636

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 22.0min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921154636/job/95101181473) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体信息，仅有Node.js版本警告和上传artifact时未找到文件的提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921154636/job/95101181473


## [Run #31921029777](https://github.com/sgl-project/sglang/actions/runs/31921029777)
- **分支**: `fix-mxfp4-sharded-state`
- **总耗时**: 50.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31921029777

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 48.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846991) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 48.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100847080) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95101291857) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 24.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95103205635) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因健康检查发现其他根因作业失败而被快速跳过，并非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95104701321) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型加载正常，但执行到TP3/TP8时容器实现执行失败，报错'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846991

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于容器环境或runner基础设施问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100847080

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1097秒后退出码1，属于性能测试未通过，可能因吞吐或延迟未达预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95101291857

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示模型分片加载到86%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95103205635

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，导致本作业未实际运行即被取消。属于依赖的上游作业失败引发的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95104701321

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846898) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846914) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846944) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846947) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846957) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846963) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100846983) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100847085) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100847088) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31921029777/job/95100847098) |


---
*Auto-generated by npu_pr_monitor.py*