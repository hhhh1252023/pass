# NPU CI 执行监控
**生成时间**: 2026-08-19 01:54 UTC
**分析 Run 数**: 56

---

## 📊 本次执行总结

- **成功 Job 数**: 272
- **失败 Run 数**: 56
- **成功 Job 平均耗时**: 21.7min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 117.8min | #32153734373 | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765903131) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.8min | #32167618314 | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818632681) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 95.6min | #32184150804 | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400667) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.1min | #32166635316 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807781498) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.9min | #32177331701 | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277985) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.4min | #32174081582 | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832921458) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.1min | #32166009777 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805764766) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.3min | #32166937173 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808812440) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.9min | #32160349439 | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789128077) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.5min | #32180304861 | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782507) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 55 | 0 | 47 | 0 | 8 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 45 | 11 | 2 | 29 | 3 |
| base-b-test-16-npu-a3 / run (0) | 45 | 14 | 1 | 25 | 5 |
| base-b-test-4-npu-a3 / run (0) | 45 | 18 | 0 | 24 | 3 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 45 | 19 | 2 | 21 | 3 |
| base-b-test-1-npu-a3 / run (0) | 45 | 20 | 0 | 22 | 3 |
| base-b-test-2-npu-a3 / run (0) | 46 | 22 | 1 | 20 | 3 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 23 | 0 | 0 | 21 | 2 |
| base-b-test-4-npu-a3 / run (1) | 46 | 25 | 1 | 17 | 3 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 45 | 23 | 2 | 14 | 6 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 28 | 12 | 0 | 14 | 2 |
| base-b-test-8-npu-a3 / run (0) | 46 | 29 | 0 | 12 | 5 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 46 | 29 | 2 | 10 | 5 |
| base-a-test-1-npu-a2 / run (0) | 46 | 35 | 1 | 10 | 0 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 11 | 0 | 0 | 11 | 0 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19 | 9 | 0 | 10 | 0 |
| multimodal-gen-test-2-npu-a3 | 1 | 0 | 1 | 0 | 0 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32166937173<br>[#35364 [PD] Fall back to cpu_tensor when host-pool retraction cannot fit](https://github.com/sgl-project/sglang/pull/35364) | `mmangkad/pd-decode-retraction-host-pool-capacity-fallback` | 352.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32166937173) |
| #32166635316<br>[#34012 Add Agentic-Aware Tail-Optimized LRU eviction to the unified radix cache](https://github.com/sgl-project/sglang/pull/34012) | `dev-dsv4-gb300-tlru` | 349.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32166635316) |
| #32166009777<br>[#34140 [AMD] [Spec] Enable stochastic tree verification on ROCm](https://github.com/sgl-project/sglang/pull/34140) | `RM/fix-dsa-paged-tree-relocation` | 333.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32166009777) |
| #32168566803<br>[#35106 [Spec] Drop the unconditional stream sync from spec-prefill H2D copies](https://github.com/sgl-project/sglang/pull/35106) | `perf/prefill-nonblocking-h2d` | 330.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32168566803) |
| #32177331701<br>[#31820 [Do Not Merge][BCG]Use piecewise cuda graphs](https://github.com/sgl-project/sglang/pull/31820) | `use-piecewise-cuda-graphs` | 188.3min | 2 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32177331701) |
| #32153734373<br>[#33602 [AMD] [GLM5] Add opt-in PTPC FP8 projections on gfx950](https://github.com/sgl-project/sglang/pull/33602) | `RM/glm52-ptpc-fp8-proj` | 131.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32153734373) |
| #32190273484<br>[#35371 [Spec] DFlash2: local convolution + candidate selector](https://github.com/sgl-project/sglang/pull/35371) | `subsir/dflash2-selector-conv` | 117.7min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32190273484) |
| #32163536232<br>[#35130 Fix NIXL cleaner grouping for hybrid cache keys](https://github.com/sgl-project/sglang/pull/35130) | `user/yawei_microsoft/fix-nixl-hybrid-cleaner-grouping` | 114.1min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32163536232) |
| #32167618314<br>[#35360 [PD] Deferred decode-side KV release for the NIXL backend](https://github.com/sgl-project/sglang/pull/35360) | `feat/nixl-deferred-decode-kv-release` | 109.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32167618314) |
| #32174081582 | `pd/abort-on-waiting-timeout` | 109.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32174081582) |
| #32184150804<br>[#34457 [PD] Enforce the request waiting timeout in disaggregation mode](https://github.com/sgl-project/sglang/pull/34457) | `pd/abort-on-waiting-timeout` | 106.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32184150804) |
| #32160349439<br>[#35318 [Perf] PaddleOCR-VL: overlap page preprocessing, pack the ViT, enable prefill CUDA graph](https://github.com/sgl-project/sglang/pull/35318) | `claude/paddleocr-support-optimization-bfaf52` | 96.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32160349439) |
| #32180304861<br>[#35372 [Kernel] Support wider rows in mega_moe_pre_dispatch](https://github.com/sgl-project/sglang/pull/35372) | `support-wide-mega-moe-pre-dispatch` | 86.7min | 1 | base-b-test-2-npu-a3 / run (0) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32180304861) |
| #32154977313<br>[#35290 [XPU] Lazily import tvm_ffi-dependent all_reduce kernel in minimax_m2](https://github.com/sgl-project/sglang/pull/35290) | `tvm_ffi` | 76.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32154977313) |
| #32190389398<br>[#34679 fix(constrained): reject NUL bytes in grammar specs to stop an xgrammar segfault](https://github.com/sgl-project/sglang/pull/34679) | `junshen/fix-nul-grammar-segfault` | 74.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32190389398) |
| #32154625018<br>[#35349 [VLM] Default to two multimodal preprocessing workers](https://github.com/sgl-project/sglang/pull/35349) | `claude/mm-processor-concurrency-default` | 59.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32154625018) |
| #32165942156<br>[#35061 [Fix] Select custom all-reduce v2 by topology capability](https://github.com/sgl-project/sglang/pull/35061) | `main` | 58.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32165942156) |
| #32183782894<br>[#35362 Laguna: config-driven MoE router scoring](https://github.com/sgl-project/sglang/pull/35362) | `main` | 57.7min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32183782894) |
| #32192932245<br>[#34923 Apply latest DeepEP branch](https://github.com/sgl-project/sglang/pull/34923) | `codex/deepep-nvshmem-qp-depth` | 56.2min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32192932245) |
| #32173758760<br>[#35130 Fix NIXL cleaner grouping for hybrid cache keys](https://github.com/sgl-project/sglang/pull/35130) | `main` | 52.3min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32173758760) |
| #32196063347<br>[#35182 fix(diffusion): reject unsupported ModelOpt checkpoint algorithms](https://github.com/sgl-project/sglang/pull/35182) | `codex/diffusion-modelopt-exact-dispatch` | 44.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32196063347) |
| #32185432733<br>[#35382 [Refactor] Share the page-aligned decode alloc lens between EAGLE and DFLASH](https://github.com/sgl-project/sglang/pull/35382) | `lsyin/share-paged-watermark-loop` | 40.4min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32185432733) |
| #32157335285<br>[#35318 [Perf] PaddleOCR-VL: overlap page preprocessing, pack the ViT, enable prefill CUDA graph](https://github.com/sgl-project/sglang/pull/35318) | `claude/paddleocr-support-optimization-bfaf52` | 37.1min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32157335285) |
| #32171106386<br>[#35248 [Metrics] Discount queued prefill load by recent cache hits when waiting-queue matching is off](https://github.com/sgl-project/sglang/pull/35248) | `main` | 33.7min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32171106386) |
| #32188878314<br>[#35382 [Refactor] Share the page-aligned decode alloc lens between EAGLE and DFLASH](https://github.com/sgl-project/sglang/pull/35382) | `main` | 33.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32188878314) |
| #32154231117<br>[#35318 [Perf] PaddleOCR-VL: overlap page preprocessing, pack the ViT, enable prefill CUDA graph](https://github.com/sgl-project/sglang/pull/35318) | `claude/paddleocr-support-optimization-bfaf52` | 32.0min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32154231117) |
| #32175052594<br>[#35184 fix(diffusion): route quantized VAE component repos safely](https://github.com/sgl-project/sglang/pull/35184) | `codex/vae-quantized-repo-routing` | 31.9min | 2 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1) | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32175052594) |
| #32194430592<br>[#35396 [Fix] Assert the page-aligned SWA evict floor on both PD decode prealloc paths](https://github.com/sgl-project/sglang/pull/35396) | `main` | 31.2min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32194430592) |
| #32176324120<br>[#35318 [Perf] PaddleOCR-VL: overlap page preprocessing, pack the ViT, enable prefill CUDA graph](https://github.com/sgl-project/sglang/pull/35318) | `claude/paddleocr-support-optimization-bfaf52` | 26.8min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32176324120) |
| #32178178842<br>[#34881 Stop losing Kimi-K3 tool calls to reasoning, constraint conflicts, and truncation](https://github.com/sgl-project/sglang/pull/34881) | `main` | 25.2min | 2 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32178178842) |
| #32195701022<br>[#35335 [diffusion] Warmup-calibrated auto residency promotion in performance-mode auto](https://github.com/sgl-project/sglang/pull/35335) | `mick/diffusion-auto-residency` | 24.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32195701022) |
| #32179752649<br>[#33778 Avoid materializing GDN QKV tensors during target verification](https://github.com/sgl-project/sglang/pull/33778) | `perf/gdn-strided-target-verify` | 23.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32179752649) |
| #32185829115<br>[#35182 fix(diffusion): reject unsupported ModelOpt checkpoint algorithms](https://github.com/sgl-project/sglang/pull/35182) | `codex/diffusion-modelopt-exact-dispatch` | 20.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32185829115) |
| #32159380581<br>[#34713 [diffusion] Decouple encoder parallelism from the DiT parallel layout](https://github.com/sgl-project/sglang/pull/34713) | `main` | 17.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32159380581) |
| #32176129512<br>[#34855 [NPU] [Diffusion] Fix NPU Ring Attention varlen dispatch & restore 2-NPU CI testcase](https://github.com/sgl-project/sglang/pull/34855) | `fix_ring_attention_npu` | 17.3min | 6 | multimodal-gen-test-1-npu-a3, multimodal-gen-test-2-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32176129512) |
| #32163916599<br>[#34627 fix: preserve output logprobs without input logprobs](https://github.com/sgl-project/sglang/pull/34627) | `main` | 16.5min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32163916599) |
| #32193214167<br>[#35396 [Fix] Assert the page-aligned SWA evict floor on both PD decode prealloc paths](https://github.com/sgl-project/sglang/pull/35396) | `lsyin/fix-hisparse-swa-fixture` | 13.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32193214167) |
| #32189052617<br>[#35371 [Spec] DFlash2: local convolution + candidate selector](https://github.com/sgl-project/sglang/pull/35371) | `subsir/dflash2-selector-conv` | 12.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32189052617) |
| #32197067402<br>[#34923 Apply latest DeepEP branch](https://github.com/sgl-project/sglang/pull/34923) | `main` | 12.4min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32197067402) |
| #32175172332<br>[#35174 [Diffusion] Reuse shared checkpoint quant metadata resolver](https://github.com/sgl-project/sglang/pull/35174) | `codex/diffusion-use-checkpoint-quant-spec` | 12.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32175172332) |
| #32165069073<br>[#35164 Refactor kv cache event mixin into a recorder](https://github.com/sgl-project/sglang/pull/35164) | `main` | 11.2min | 3 | multimodal-gen-test-1-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32165069073) |
| #32158423544<br>[#30319 [NPU] Add mxfp4-w4a4 MOE Quantization Support for NPU](https://github.com/sgl-project/sglang/pull/30319) | `main` | 10.9min | 5 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32158423544) |
| #32176475612<br>[#35182 fix(diffusion): reject unsupported ModelOpt checkpoint algorithms](https://github.com/sgl-project/sglang/pull/35182) | `codex/diffusion-modelopt-exact-dispatch` | 10.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32176475612) |
| #32157490647<br>[#35049 [PD] Deferred decode-side KV release for aborts mid-transfer](https://github.com/sgl-project/sglang/pull/35049) | `main` | 10.7min | 4 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32157490647) |
| #32175114286<br>[#35182 fix(diffusion): reject unsupported ModelOpt checkpoint algorithms](https://github.com/sgl-project/sglang/pull/35182) | `codex/diffusion-modelopt-exact-dispatch` | 9.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32175114286) |
| #32158668938<br>[#35352 [diffusion] ComfyUI: add a MiniMax-H3 node and a generic extra-fields passthrough](https://github.com/sgl-project/sglang/pull/35352) | `codex/comfyui-minimax-h3-node` | 9.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32158668938) |
| #32156721326<br>[#35238 Exclude multimodal-gen NPU jobs from fast-fail cascade](https://github.com/sgl-project/sglang/pull/35238) | `main` | 9.0min | 3 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32156721326) |
| #32158717122<br>[#35353 [diffusion] make --vae-tiling honest, fix the decode OOM advice, gate NVFP4 on Blackwell](https://github.com/sgl-project/sglang/pull/35353) | `codex/vae-tiling-and-nvfp4-gate` | 8.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32158717122) |
| #32175384720<br>[#30004 [diffusion] feat: per-layer TP shard planner for DiT linears (--dit-tp-plan)](https://github.com/sgl-project/sglang/pull/30004) | `mick/dit-tp-shard-planner` | 8.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32175384720) |
| #32160109975<br>[#35339 [diffusion] Per-request lossy accelerations: Cache-DiT, CFG gating, attention backend override](https://github.com/sgl-project/sglang/pull/35339) | `claude/dynamic-per-request-config-20760a` | 8.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32160109975) |
| #32163650848<br>[#35335 [diffusion] Warmup-calibrated auto residency promotion in performance-mode auto](https://github.com/sgl-project/sglang/pull/35335) | `mick/diffusion-auto-residency` | 8.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32163650848) |
| #32174861977<br>[#35335 [diffusion] Warmup-calibrated auto residency promotion in performance-mode auto](https://github.com/sgl-project/sglang/pull/35335) | `mick/diffusion-auto-residency` | 7.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32174861977) |
| #32183157660<br>[#35265 [Spec] Page-align the DFLASH decode KV reservation](https://github.com/sgl-project/sglang/pull/35265) | `main` | 7.6min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32183157660) |
| #32163799367<br>[#34893 [Diffusion] Add MiniMax H3 cube sparse attention](https://github.com/sgl-project/sglang/pull/34893) | `codex/minimax-h3-cube-sparse-attn` | 6.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32163799367) |
| #32175797035<br>[#35353 [diffusion] make --vae-tiling honest, fix the decode OOM advice, gate NVFP4 on Blackwell](https://github.com/sgl-project/sglang/pull/35353) | `codex/vae-tiling-and-nvfp4-gate` | 6.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32175797035) |
| #32196698986<br>[#35375 [Memory] Borrow CUDA graph pool storage for EAGLE sampling](https://github.com/sgl-project/sglang/pull/35375) | `main` | 5.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32196698986) |

---


## [Run #32197067402](https://github.com/sgl-project/sglang/actions/runs/32197067402)
- **分支**: `main`
- **总耗时**: 12.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32197067402

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270519) |
| base-b-test-4-npu-a3 / run (1) | 6.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270639) |
| base-b-test-4-npu-a3 / run (0) | 4.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270677) |
| base-b-test-1-npu-a3 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270695) |
| base-b-test-8-npu-a3 / run (0) | 3.9min | 环境问题 | 自定义容器启动失败，导致作业无法运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270722) |
| base-b-test-2-npu-a3 / run (0) | 4.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270746) |
| base-b-test-16-npu-a3 / run (0) | 4.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270767) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | 其他 | 测试套件未找到任何测试用例，属于预期跳过，但脚本因退出码解析错误导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270865) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 2.0min | 环境问题 | 自定义容器执行失败，依赖安装过程中环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270922) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.6min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270943) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.2min | 环境问题 | 自定义容器执行失败，导致测试未开始即中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903271078) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270519

- **base-b-test-4-npu-a3 / run (1)**: 日志显示服务已成功启动并完成请求处理，但随后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270639

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载模型分片后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270677

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270695

- **base-b-test-8-npu-a3 / run (0)**: 日志显示容器初始化过程中出现torch_npu相关警告，随后报错“Executing the custom container implementation failed”，说明NPU容器环境配置或启动存在问题，并非测试代码本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270722

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重时（Multi-thread loading shards 0%）容器实现执行失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270746

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动NPU容器后，TokenizerManager初始化过程中容器执行失败，报错'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270767

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示“No tests found for hw=NPU, suite=base-c-test-acc-8-npu-a3”，测试被跳过。随后shell脚本中`[: 0\n0: integer expression expected`报错，说明退出码处理逻辑有误，导致作业被标记为失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270865

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 在安装triton-ascend等依赖时，卸载并重装numpy、scipy等包后，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270922

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装triton-ascend依赖时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270943

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在运行测试前，执行自定义容器实现时失败，错误为“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于基础设施或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903271078

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32197067402/job/95903270759) |


## [Run #32196698986](https://github.com/sgl-project/sglang/actions/runs/32196698986)
- **分支**: `main`
- **总耗时**: 5.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32196698986

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244628) |
| base-b-test-16-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244647) |
| base-a-test-1-npu-a2 / run (0) | 4.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244652) |
| base-b-test-4-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244680) |
| base-b-test-4-npu-a3 / run (1) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244700) |
| base-b-test-8-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244706) |
| base-b-test-1-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244728) |
| base-b-test-2-npu-a3 / run (0) | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244747) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902245087) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902245145) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902245186) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902245257) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明作业依赖的某个文件或数据在存储账户中缺失，可能是上传失败、路径错误或资源被清理，需检查CI配置中的blob引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244628

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244647

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动自定义容器时失败，错误信息为"Executing the custom container implementation failed"，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244652

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244680

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244700

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244706

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成或存储配置有误，需检查相关 blob 路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244728

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902244747

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902245087

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902245145

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902245186

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置变更导致，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196698986/job/95902245257


## [Run #32196063347](https://github.com/sgl-project/sglang/actions/runs/32196063347)
- **分支**: `codex/diffusion-modelopt-exact-dispatch`
- **总耗时**: 44.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32196063347

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 作业未显示实际测试失败，仅上传失败产物目录为空，可能测试未执行或已通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305363) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305430) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305462) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305470) |
| base-b-test-8-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305503) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现根因任务multimodal-gen-test-1-npu-a3失败，触发级联失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305524) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他作业根因失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305540) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305886) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305960) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95901957930) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.7min | 其他 | 健康检查快速失败，根因作业为multimodal-gen-test-1-npu-a3 | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95905808421) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试命令或失败断言，仅显示上传diffusion-failures目录时无文件，说明该目录为空或未生成，作业可能因无失败产物而正常结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305363

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305430

- **base-b-test-1-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305462

- **base-b-test-16-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而被级联跳过，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305470

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而快速失败，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305503

- **base-b-test-4-npu-a3 / run (0)**: 本作业因PR测试健康检查检测到根因任务multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，属于级联失败而非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305524

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305540

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 该作业在启动前的PR测试健康检查中，因检测到根因作业multimodal-gen-test-1-npu-a3失败，触发了fast-fail机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305886

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动前的健康检查阶段检测到同一PR中的multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业被跳过未实际执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305960

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95901957930

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业因PR健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而触发快速失败机制，被跳过执行，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95905808421

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305502) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900305962) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32196063347/job/95900306004) |


## [Run #32195701022](https://github.com/sgl-project/sglang/actions/runs/32195701022)
- **分支**: `mick/diffusion-auto-residency`
- **总耗时**: 24.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32195701022

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32195701022/job/95899232998) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的实际执行输出或失败原因，可能因日志截断或作业在测试前已终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32195701022/job/95899232998


## [Run #32194430592](https://github.com/sgl-project/sglang/actions/runs/32194430592)
- **分支**: `main`
- **总耗时**: 31.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32194430592

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492304) |
| base-b-test-1-npu-a3 / run (0) | 29.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492400) |
| base-b-test-4-npu-a3 / run (0) | 7.9min | 代码错误 | HiCache MLA测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492488) |
| base-b-test-4-npu-a3 / run (1) | 0.7min | 环境问题 | 健康检查发现根因作业失败，触发快速失败机制，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492542) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492556) |
| base-b-test-2-npu-a3 / run (0) | 19.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492619) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492928) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.5min | 其他 | 作业因其他根因作业失败而被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492963) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 21.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895493052) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95897373540) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未包含multimodal测试的具体执行结果或错误输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492304

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中（84%进度）时，自托管runner报告"Executing the custom container implementation failed"，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492400

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试执行失败，退出码为1，导致整个作业失败。测试耗时281秒，未通过，具体失败原因需查看该测试文件的详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492488

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因作业 base-b-test-4-npu-a3 / run (0) 失败，随后触发 fast-fail 跳过，导致当前作业未实际运行即退出。根因可能是环境或前置依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492542

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业 base-b-test-4-npu-a3 / run (0) 失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492556

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试运行至73%时，自托管runner报错“Executing the custom container implementation failed”，随后进入清理流程，属于runner或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492619

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492928

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 健康检查发现根因作业 base-b-test-4-npu-a3 / run (0) 失败，触发 fast-fail 机制，导致本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492963

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（解码吞吐约620 token/s），但突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895493052

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动前的PR健康检查中，检测到根因作业base-b-test-4-npu-a3/run(0)失败，触发了fast-fail机制，本作业未实际执行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95897373540

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 11.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492482) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895492630) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32194430592/job/95895493078) |


## [Run #32193214167](https://github.com/sgl-project/sglang/actions/runs/32193214167)
- **分支**: `lsyin/fix-hisparse-swa-fixture`
- **总耗时**: 13.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32193214167

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498648) |
| base-b-test-4-npu-a3 / run (0) | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498656) |
| base-b-test-4-npu-a3 / run (1) | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498657) |
| base-b-test-2-npu-a3 / run (0) | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498689) |
| base-b-test-16-npu-a3 / run (0) | 13.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498692) |
| base-b-test-8-npu-a3 / run (0) | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498747) |
| base-b-test-1-npu-a3 / run (0) | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498802) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498950) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498970) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498981) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 13.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892499032) |

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是资源被删除、路径错误或未上传成功，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498648

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498656

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498657

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或资源配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498689

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498692

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498747

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498802

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置变更导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498950

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498970

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498981

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892499032

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32193214167/job/95892498849) |


## [Run #32192932245](https://github.com/sgl-project/sglang/actions/runs/32192932245)
- **分支**: `codex/deepep-nvshmem-qp-depth`
- **总耗时**: 56.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32192932245

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077587) |
| base-b-test-16-npu-a3 / run (0) | 55.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077786) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 52.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891078264) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95892291609) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 10.6min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95900244185) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95901987372) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures目录（无文件），未包含multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077587

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试运行正常，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077786

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891078264

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在性能测试运行过程中，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95892291609

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示测试运行正常，但在执行过程中出现错误：Executing the custom container implementation failed，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95900244185

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95901987372

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077737) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077751) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077821) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077866) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077887) |
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891077893) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891078314) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891078367) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32192932245/job/95891078440) |


## [Run #32190389398](https://github.com/sgl-project/sglang/actions/runs/32190389398)
- **分支**: `junshen/fix-nul-grammar-segfault`
- **总耗时**: 74.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32190389398

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.6min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志中无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444020) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查失败：lint检查未通过导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444297) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 环境问题 | 健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444393) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查失败：lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444703) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.7min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95885762779) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 健康检查失败：lint检查未通过导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95890190959) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现lint检查失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95892899638) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures步骤，但未找到任何失败文件，且无测试执行或报错记录，可能因前置条件未满足或作业被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444020

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到PR的lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444297

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发fast-fail机制，作业被终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444393

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444703

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查状态为failure，触发fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95885762779

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95890190959

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行健康检查时，检测到PR的lint检查结论为failure，触发了fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95892899638

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883222283) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444165) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444167) |
| base-b-test-16-npu-a3 / run (0) | 52.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444204) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444263) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444280) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444602) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444647) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190389398/job/95883444725) |


## [Run #32190273484](https://github.com/sgl-project/sglang/actions/runs/32190273484)
- **分支**: `subsir/dflash2-selector-conv`
- **总耗时**: 117.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32190273484

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883231791) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 66.2min | 环境问题 | NPU服务器启动超时，服务未就绪导致测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95889133561) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 23.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95902392943) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时无文件，可能测试通过或未生成失败产物，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883231791

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: SGLang服务在60秒内未能启动，服务器未就绪，随后自定义容器执行失败，属于环境或启动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95889133561

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示测试运行正常，但执行自定义容器时失败，提示联系自托管runner管理员，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95902392943

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232020) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232027) |
| base-b-test-16-npu-a3 / run (0) | 52.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232046) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232138) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232147) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232170) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232199) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 80.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232526) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232561) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232643) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95883232713) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95884121740) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32190273484/job/95893113696) |


## [Run #32189052617](https://github.com/sgl-project/sglang/actions/runs/32189052617)
- **分支**: `subsir/dflash2-selector-conv`
- **总耗时**: 12.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32189052617

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 3.6min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309130) |
| base-b-test-4-npu-a3 / run (1) | 3.5min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309160) |
| base-a-test-1-npu-a2 / run (0) | 4.5min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309164) |
| base-b-test-8-npu-a3 / run (0) | 3.3min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309188) |
| base-b-test-4-npu-a3 / run (0) | 3.0min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309192) |
| base-b-test-1-npu-a3 / run (0) | 3.7min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309212) |
| base-b-test-2-npu-a3 / run (0) | 4.2min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309227) |
| multimodal-gen-test-1-npu-a3 | 2.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309228) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.5min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309582) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.5min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309594) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.8min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309611) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.2min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309623) |

- **base-b-test-16-npu-a3 / run (0)**: test_spec_aux_hidden_state.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，collect_tests 抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309130

- **base-b-test-4-npu-a3 / run (1)**: test/registered/unit/spec/test_dflash_logits.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，collect_tests 抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309160

- **base-a-test-1-npu-a2 / run (0)**: test_spec_aux_hidden_state.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，collect_tests 抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309164

- **base-b-test-8-npu-a3 / run (0)**: test/registered/unit/spec/test_dflash_logits.py缺少`if __name__ == "__main__":`入口，导致pytest风格测试在`python3 file.py -f`下静默跳过，run_suite.py抛出ValueError异常，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309188

- **base-b-test-4-npu-a3 / run (0)**: test/registered/unit/spec/test_dflash_logits.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 python3 file.py -f 下静默跳过，collect_tests 抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309192

- **base-b-test-1-npu-a3 / run (0)**: test/registered/unit/spec/test_dflash_logits.py缺少`if __name__ == "__main__":`入口，导致pytest风格测试在`python3 file.py -f`下静默跳过，run_suite.py抛出ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309212

- **base-b-test-2-npu-a3 / run (0)**: test/registered/unit/spec/test_dflash_logits.py缺少`if __name__ == "__main__":`入口，导致pytest风格测试在`python3 file.py -f`下静默跳过，run_suite.py抛出ValueError异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309227

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤。可能因日志截断或作业在测试前被取消，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309228

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: test/registered/unit/spec/test_dflash_logits.py缺少`if __name__ == "__main__":`入口，pytest风格测试在`python3 file.py -f`下会静默跳过，需添加`sys.exit(pytest.main([__file__, "-v"]))`。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309582

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: test/registered/unit/spec/test_dflash_logits.py缺少`if __name__ == "__main__":`入口，导致pytest风格测试在`python3 file.py -f`下静默跳过，collect_tests抛出ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309594

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: test/registered/unit/spec/test_dflash_logits.py缺少`if __name__ == "__main__":`入口，导致pytest风格测试在`python3 file.py -f`下静默跳过，collect_tests抛出ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309611

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: test_spec_aux_hidden_state.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，CI 收集测试时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/32189052617/job/95879309623


## [Run #32188878314](https://github.com/sgl-project/sglang/actions/runs/32188878314)
- **分支**: `main`
- **总耗时**: 33.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32188878314

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.9min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060497) |
| base-b-test-16-npu-a3 / run (0) | 22.5min | 环境问题 | NPU容器执行失败，自定义容器实现报错 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060668) |
| base-b-test-4-npu-a3 / run (1) | 22.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060670) |
| base-b-test-4-npu-a3 / run (0) | 7.9min | 代码错误 | HiCache MLA 测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060687) |
| base-b-test-1-npu-a3 / run (0) | 25.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060743) |
| base-b-test-2-npu-a3 / run (0) | 23.3min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060931) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 25.8min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879061084) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879061141) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.4min | 其他 | 作业实际成功，日志显示测试通过，无失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95881189268) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95885439144) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures目录，但未找到任何文件，未运行实际测试或输出失败原因，可能因前置条件未满足或作业被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060497

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载MoE模型权重时发生torch copy_操作错误，随后Scheduler watchdog超时，最终自定义容器执行失败，可能是NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060668

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060670

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py 测试文件运行失败（exit code 1），耗时281秒，导致整个作业失败。具体失败原因需查看该测试的详细输出，可能是测试断言失败或运行时错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060687

- **base-b-test-1-npu-a3 / run (0)**: 日志显示模型权重加载成功，但在后续执行阶段出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于运行环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060743

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在TokenizerManager初始化后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060931

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示服务正常运行，但执行到22:07:48时出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879061084

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在22:07:42出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879061141

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试全部通过（1/1 passed），吞吐量394.45高于基线390.59，作业正常结束。可能为误报或后续步骤问题，但当前日志无失败迹象。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95881189268

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到base-b-test-4-npu-a3作业失败，作为根因作业触发fast-fail，导致本作业未实际运行即被终止，属于CI环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95885439144

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95878811664) |
| base-b-test-8-npu-a3 / run (0) | 10.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060658) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879060775) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879061184) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32188878314/job/95879061261) |


## [Run #32185829115](https://github.com/sgl-project/sglang/actions/runs/32185829115)
- **分支**: `codex/diffusion-modelopt-exact-dispatch`
- **总耗时**: 20.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32185829115

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 作业未显示实际测试失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175191) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175279) |
| base-b-test-4-npu-a3 / run (1) | 1.1min | 其他 | 级联失败：因其他根因作业失败被快速跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175302) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175309) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | 环境问题 | NPU测试用例执行失败，报未知应用异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175315) |
| base-b-test-1-npu-a3 / run (0) | 1.2min | 其他 | 健康检查快速失败，根因作业在其他任务中失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175348) |
| base-b-test-16-npu-a3 / run (0) | 1.2min | 其他 | 作业因级联失败被快速跳过，根因是其他作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175358) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175409) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 2.0min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175781) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175827) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | PR健康检查失败，因其他根因作业失败导致本作业被快速跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175837) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 作业因其他根因作业失败被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175946) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行细节，仅显示上传diffusion-failures目录时无文件，可能测试未运行或未产生失败样本，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175191

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动前的健康检查中检测到multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2两个根因作业失败，按策略快速失败，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175279

- **base-b-test-4-npu-a3 / run (1)**: 该作业本身未执行测试，因健康检查发现根因作业（multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2）失败，触发fast-fail机制被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175302

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查过滤掉多个级联失败作业后，根因失败为multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2/run(0)，导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175309

- **base-a-test-1-npu-a2 / run (0)**: test_npu_ascend_backend.py在NPU上运行时报ERR99999 UNKNOWN application exception，测试0/2通过，可能是NPU环境或依赖问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175315

- **base-b-test-1-npu-a3 / run (0)**: 该作业因PR健康检查检测到根因作业（multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2）失败而被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175348

- **base-b-test-16-npu-a3 / run (0)**: 该作业在健康检查阶段检测到根因作业（multimodal-gen-test-1-npu-a3 和 base-a-test-1-npu-a2）失败，触发 fast-fail 机制，导致本作业被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175358

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动阶段被健康检查拦截，检测到multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2两个根因作业失败，触发fast-fail机制，本作业未实际执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175409

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175781

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查过滤掉级联失败后，根因失败作业为multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2/run(0)，本作业因这些根因失败被快速失败机制跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175827

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2/run(0)，本作业因级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175837

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示该作业被健康检查过滤为级联失败，根因是multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2作业失败，导致本作业被快速失败跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185829115/job/95869175946


## [Run #32185432733](https://github.com/sgl-project/sglang/actions/runs/32185432733)
- **分支**: `lsyin/share-paged-watermark-loop`
- **总耗时**: 40.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32185432733

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含GitHub Actions基础设施警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869049912) |
| base-b-test-4-npu-a3 / run (0) | 35.4min | 其他 | 作业实际测试全部通过，失败可能由基础设施或日志截断导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050323) |
| base-b-test-16-npu-a3 / run (0) | 26.2min | 环境问题 | 自定义容器执行失败，服务启动后容器异常退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050482) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 31.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869051148) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 31.3min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869051161) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.6min | 环境问题 | 自托管runner执行容器失败，作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95870619867) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 6.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95876943157) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含runner启动、Node 20弃用警告及上传artifact时未找到diffusion-failures目录的提示，未展示测试执行过程或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869049912

- **base-b-test-4-npu-a3 / run (0)**: 日志显示5个NPU测试全部PASSED，无测试失败或超时。作业失败可能源于runner清理阶段或基础设施问题，而非代码或测试本身。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050323

- **base-b-test-16-npu-a3 / run (0)**: 日志显示服务已成功启动并完成预热，但随后出现"Executing the custom container implementation failed"错误，可能是容器环境或资源限制导致进程被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050482

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在21:40:25时出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869051148

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然报错“Executing the custom container implementation failed”，随后作业终止，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869051161

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示性能测试运行正常，但随后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner环境或容器执行问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95870619867

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示模型权重加载到59%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95876943157

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050149) |
| base-b-test-2-npu-a3 / run (0) | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050189) |
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050200) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050227) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869050273) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869051122) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32185432733/job/95869051333) |


## [Run #32184150804](https://github.com/sgl-project/sglang/actions/runs/32184150804)
- **分支**: `pd/abort-on-waiting-timeout`
- **总耗时**: 106.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32184150804

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.3min | 其他 | 作业日志不完整，未显示实际测试执行过程，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865399831) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95867324241) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因同PR中另一个作业失败被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95875823552) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.5min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95875978993) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95890871989) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有runner初始化、action下载和上传artifact步骤，未包含multimodal测试的实际运行输出。上传diffusion-failures时提示无文件，说明测试可能未执行或未产生失败文件，但无法从日志确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865399831

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1125秒后失败，0/1通过，属于性能测试未达标，可能因模型推理延迟或吞吐量未满足50ms目标导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95867324241

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动前被健康检查机制拦截，原因是同PR中的base-c-test-perf-8-npu-a3作业已失败，触发fast-fail逻辑，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95875823552

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业在启动阶段被PR健康检查机制快速失败，原因是同批次中base-c-test-perf-8-npu-a3作业失败被判定为根因，本作业作为级联失败被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95875978993

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95890871989

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865399958) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865399978) |
| base-b-test-4-npu-a3 / run (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400012) |
| base-b-test-8-npu-a3 / run (0) | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400075) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400091) |
| base-b-test-16-npu-a3 / run (0) | 46.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400092) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400189) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 37.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400570) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400600) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 95.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400667) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32184150804/job/95865400780) |


## [Run #32183782894](https://github.com/sgl-project/sglang/actions/runs/32183782894)
- **分支**: `main`
- **总耗时**: 57.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32183782894

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070454) |
| base-b-test-16-npu-a3 / run (0) | 53.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070745) |
| base-b-test-4-npu-a3 / run (0) | 8.3min | 代码错误 | HiCache MLA测试文件执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863071077) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 55.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863071930) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95870965125) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95874486334) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能确认上传diffusion-failures目录时无文件，说明测试可能未产生失败产物或提前退出，需查看完整日志定位具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070454

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在加载模型分片时出现“Executing the custom container implementation failed”错误，属于自托管运行器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070745

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试用例运行291秒后失败，0/5测试通过，具体错误信息被截断，但可确定是该测试文件本身存在问题导致执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863071077

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863071930

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-b-test-4-npu-a3作业失败，被判定为根因作业，因此本作业（base-c-test-perf-16-npu-a3）被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95870965125

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行PR测试健康检查时，检测到根因作业base-b-test-4-npu-a3失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95874486334

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070805) |
| base-b-test-8-npu-a3 / run (0) | 11.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070813) |
| base-b-test-4-npu-a3 / run (1) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070923) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070956) |
| base-b-test-1-npu-a3 / run (0) | 46.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863070975) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863071796) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863071810) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95863071835) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183782894/job/95864705767) |


## [Run #32183157660](https://github.com/sgl-project/sglang/actions/runs/32183157660)
- **分支**: `main`
- **总耗时**: 7.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32183157660

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.8min | 其他 | 日志不完整，未显示测试执行过程，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861126669) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861126931) |
| base-b-test-2-npu-a3 / run (0) | 5.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127070) |
| base-b-test-8-npu-a3 / run (0) | 5.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127124) |
| base-b-test-4-npu-a3 / run (0) | 4.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127135) |
| base-b-test-4-npu-a3 / run (1) | 4.4min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，且自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127170) |
| base-b-test-1-npu-a3 / run (0) | 5.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127201) |
| base-b-test-16-npu-a3 / run (0) | 5.2min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127224) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.3min | 环境问题 | 自定义容器执行失败，模型加载中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127689) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127777) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.2min | 环境问题 | 自定义容器执行失败，导致测试未真正运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127818) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95862500298) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions初始化、Node版本警告及上传diffusion-failures目录（无文件）的步骤，未展示实际测试命令或错误输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861126669

- **base-a-test-1-npu-a2 / run (0)**: 作业在运行NPU测试时，自定义容器实现执行失败，提示联系自托管runner管理员。可能是容器镜像或NPU驱动环境配置问题，导致测试无法正常启动。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861126931

- **base-b-test-2-npu-a3 / run (0)**: 作业在加载模型权重时（Multi-thread loading shards 0%）容器执行失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127070

- **base-b-test-8-npu-a3 / run (0)**: 日志显示在测试运行过程中，自定义容器实现执行失败，提示联系自托管runner管理员。这属于runner环境或容器配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127124

- **base-b-test-4-npu-a3 / run (0)**: 日志显示模型权重加载完成后，自定义容器实现执行失败（Executing the custom container implementation failed），可能是NPU设备或容器环境配置问题，导致作业在启动阶段崩溃。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127135

- **base-b-test-4-npu-a3 / run (1)**: 日志显示下载 actions/checkout 时返回 429 Too Many Requests，重试后仍失败，随后自定义容器实现执行报错，导致作业在测试开始前中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127170

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行中容器突然崩溃，报错'Executing the custom container implementation failed'，提示联系runner管理员，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127201

- **base-b-test-16-npu-a3 / run (0)**: 日志显示服务启动正常，但随后出现"Executing the custom container implementation failed"错误，表明自托管runner的容器环境存在问题，导致测试无法继续执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127224

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载Qwen3-VL模型分片时（31%进度）报错“Executing the custom container implementation failed”，属于自托管runner容器环境异常，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127689

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型加载过程中（加载48个分片至6%时）出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于容器环境或runner配置问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127777

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试命令已开始执行，但随后报错"Executing the custom container implementation failed"，说明自托管runner的容器环境存在问题，测试未能正常完成。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127818

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95862500298

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32183157660/job/95861127876) |


## [Run #32180304861](https://github.com/sgl-project/sglang/actions/runs/32180304861)
- **分支**: `support-wide-mega-moe-pre-dispatch`
- **总耗时**: 86.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32180304861

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 1.3min | 环境问题 | GitHub Actions 下载 checkout action 时遭遇 429 限流，重试后仍失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782000) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.3min | 环境问题 | 健康检查发现其他作业失败导致本作业被快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95853613515) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.2min | 其他 | 上游作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95859094152) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95863802424) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95876513022) |

- **base-b-test-2-npu-a3 / run (0)**: 作业在准备阶段下载 actions/checkout@v4 时，codeload.github.com 返回 429 Too Many Requests，重试 3 次均失败，导致作业无法启动，属于 GitHub 服务端限流或网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782000

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业启动时健康检查检测到同批次作业base-b-test-2-npu-a3失败，触发了fast-fail机制，本作业未实际运行即被跳过，属于关联作业失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95853613515

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 本作业因根因作业base-b-test-2-npu-a3失败被健康检查快速失败机制跳过，非自身问题。日志显示actions/checkout下载曾遇429限流，但最终成功，主要失败原因为上游级联。
  链接: https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95859094152

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-2-npu-a3，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95863802424

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-2-npu-a3，本作业被判定为级联失败而快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95876513022

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851781929) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851781985) |
| base-b-test-16-npu-a3 / run (0) | 54.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782003) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782067) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782105) |
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782132) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782488) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782507) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782562) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32180304861/job/95851782713) |


## [Run #32179752649](https://github.com/sgl-project/sglang/actions/runs/32179752649)
- **分支**: `perf/gdn-strided-target-verify`
- **总耗时**: 23.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32179752649

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32179752649/job/95914241948) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node.js弃用警告和上传artifact时无文件提示，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32179752649/job/95914241948

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32179752649/job/95914242097) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32179752649/job/95914242450) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32179752649/job/95914242593) |
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32179752649/job/95914242666) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32179752649/job/95914243039) |


## [Run #32178178842](https://github.com/sgl-project/sglang/actions/runs/32178178842)
- **分支**: `main`
- **总耗时**: 25.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32178178842

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.3min | 环境问题 | GitHub Actions 下载 actions/checkout 时遭遇 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453798) |
| base-b-test-8-npu-a3 / run (0) | 5.0min | 环境问题 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453817) |
| base-b-test-16-npu-a3 / run (0) | 19.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453849) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他作业失败导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453857) |
| base-b-test-1-npu-a3 / run (0) | 1.4min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453894) |
| base-b-test-4-npu-a3 / run (0) | 8.7min | 代码错误 | HiCache MLA测试文件执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453932) |
| base-b-test-4-npu-a3 / run (1) | 18.7min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453973) |
| base-a-test-1-npu-a2 / run (0) | 1.2min | 环境问题 | GitHub Actions 下载 checkout action 时遭遇 429 限流，重试后仍失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454002) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454258) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454260) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.9min | 其他 | 健康检查快速失败，因其他作业已失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454303) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.4min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，且健康检查发现其他根因作业失败导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454462) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/checkout 时返回 429 Too Many Requests，触发重试后仍失败，属于 GitHub 服务端限流导致的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453798

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是base-a-test-1-npu-a2作业失败，本作业作为依赖被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453817

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453849

- **base-b-test-2-npu-a3 / run (0)**: 该作业在启动时被健康检查拦截，因为同批次中base-a-test-1-npu-a2作业已失败，触发了fast-fail机制，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453857

- **base-b-test-1-npu-a3 / run (0)**: 日志显示下载 actions/checkout@v4 时返回 429 Too Many Requests，重试后仍失败，属于 GitHub 服务端限流导致的环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453894

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试在283秒后失败，退出码为1，导致整个作业失败。具体失败原因需查看该测试文件的详细输出，可能是测试逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453932

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在加载模型权重时（约56%进度），出现错误：Executing the custom container implementation failed. Please contact your self hosted runner administrator。这属于自托管runner环境问题，而非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846453973

- **base-a-test-1-npu-a2 / run (0)**: 作业在准备阶段下载 actions/checkout@v4 时，因 GitHub 返回 429 Too Many Requests（请求过多）导致下载失败，重试 3 次后仍无法获取，属于 GitHub 服务端限流或网络环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454002

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到根因作业 base-a-test-1-npu-a2 / run (0) 失败，触发了 fast-fail 机制，本作业未实际运行测试即被终止，属于上游失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454258

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动前的健康检查中检测到根因作业 base-a-test-1-npu-a2 / run (0) 失败，触发了 fast-fail 机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454260

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业base-a-test-1-npu-a2/run，触发fast-fail机制，本作业未实际执行测试即被终止，属于CI流程中的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454303

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示下载 actions/checkout 时返回 429 Too Many Requests，重试后仍失败。同时健康检查识别出 base-b-test-4-npu-a3 和 base-a-test-1-npu-a2 为根因失败，本作业作为级联失败被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32178178842/job/95846454462


## [Run #32177331701](https://github.com/sgl-project/sglang/actions/runs/32177331701)
- **分支**: `use-piecewise-cuda-graphs`
- **总耗时**: 188.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32177331701

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业日志不完整，未显示实际测试命令和失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277294) |
| base-b-test-16-npu-a3 / run (0) | 1.2min | 环境问题 | GitHub Actions 下载 checkout action 时遭遇 429 限流，重试后仍失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277404) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 54.3min | 性能回归 | 性能测试用例失败，deepseek_v4_flash测试未通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95849894260) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 79.2min | 性能回归 | NPU性能测试用例失败，qwen3_6_27b_w8a8_1p_in64k_out1k_50ms测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95869342284) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件（未找到文件）等步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277294

- **base-b-test-16-npu-a3 / run (0)**: 作业在准备阶段下载 actions/checkout@v4 时，因 GitHub 返回 429 Too Many Requests（请求过多）导致下载失败，重试 3 次后仍无法获取，属于 GitHub 服务端限流或网络环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277404

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 在NPU性能测试中，deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms测试用例执行失败（exit code 1），其他三个用例均通过，表明该模型性能未达预期或存在回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95849894260

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py执行失败（exit code 1），该用例为性能测试，可能因性能未达预期或运行错误导致，其他用例均通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95869342284

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277437) |
| base-b-test-4-npu-a3 / run (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277495) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277497) |
| base-b-test-4-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277513) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277581) |
| base-b-test-2-npu-a3 / run (0) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277607) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277892) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277985) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842277990) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95842278040) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 18.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95843753140) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32177331701/job/95853126215) |


## [Run #32176475612](https://github.com/sgl-project/sglang/actions/runs/32176475612)
- **分支**: `codex/diffusion-modelopt-exact-dispatch`
- **总耗时**: 10.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32176475612

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839655663) |
| base-b-test-4-npu-a3 / run (0) | 1.3min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839655977) |
| base-b-test-16-npu-a3 / run (0) | 1.7min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656006) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，非本作业自身问题 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656105) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | 环境问题 | NPU测试因Ascend后端异常失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656111) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656128) |
| base-b-test-2-npu-a3 / run (0) | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656197) |
| base-b-test-4-npu-a3 / run (1) | 0.7min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656205) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656577) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 精度回归 | NPU精度测试用例失败，GLM5模型GSM8K测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656603) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.7min | 精度回归 | NPU精度测试用例qwen3_vl_30b_a3b_bf16_2p_gsm8k失败，0/2测试通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656740) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | PR健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656904) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（无文件上传），未包含任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839655663

- **base-b-test-4-npu-a3 / run (0)**: 本作业在“Check PR test health”步骤被快速失败机制跳过，原因是根因作业multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2失败，本作业并非实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839655977

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到 `base-a-test-1-npu-a2 / run (0)` 作业失败，被判定为根因失败，因此本作业（base-b-test-16-npu-a3）被快速失败跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656006

- **base-b-test-8-npu-a3 / run (0)**: 该作业因其他根因作业（如multimodal-gen-test-1-npu-a3等）失败而被级联跳过，属于健康检查快速失败机制触发，非本作业代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656105

- **base-a-test-1-npu-a2 / run (0)**: test_npu_ascend_backend.py测试在NPU上运行时报ERR99999未知应用异常，导致测试进程退出码1，最终作业失败。可能是NPU环境或CANN版本兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656111

- **base-b-test-1-npu-a3 / run (0)**: 本作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2两个根因作业失败，按策略快速失败，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656128

- **base-b-test-2-npu-a3 / run (0)**: 本作业在“Check PR test health”步骤被快速失败机制跳过，原因是其他根因作业（multimodal-gen-test-1-npu-a3、base-a-test-1-npu-a2）失败，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656197

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行PR健康检查时，检测到根因作业base-a-test-1-npu-a2失败，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656205

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业base-a-test-1-npu-a2/run(0)，触发fast-fail机制，本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656577

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，0/1测试通过，耗时仅14秒，疑似模型精度未达预期或运行异常，导致CI流程终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656603

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 测试test_npu_qwen3_vl_30b_a3b_bf16_2p_gsm8k.py返回退出码1，所有2个测试均未通过，属于精度回归问题，可能由模型权重、推理配置或NPU环境变化导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656740

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 本作业在启动前的PR健康检查阶段被快速失败（fast-fail），原因是同一次运行中其他作业（multimodal-gen-test-1-npu-a3和base-a-test-1-npu-a2）已失败，本作业被级联跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176475612/job/95839656904


## [Run #32176324120](https://github.com/sgl-project/sglang/actions/runs/32176324120)
- **分支**: `claude/paddleocr-support-optimization-bfaf52`
- **总耗时**: 26.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32176324120

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.7min | 环境问题 | GitHub Actions 拉取代码时，远程仓库缺少指定 commit，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175610) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175767) |
| base-b-test-4-npu-a3 / run (0) | 1.7min | 环境问题 | GitHub Actions 下载 actions/checkout 时遭遇 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175851) |
| base-b-test-16-npu-a3 / run (0) | 1.5min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175930) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.4min | 环境问题 | GitHub Actions 下载 checkout 动作时遭遇 429 限流，重试后仍失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839176229) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.7min | 环境问题 | Git 拉取代码失败，远端仓库缺少指定 commit 引用。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839176331) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.5min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，且后续 git fetch 失败，提示 ref 不存在。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839176343) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 2.9min | 环境问题 | GitHub Actions 无法获取 PR 合并后的提交引用，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839176369) |

- **multimodal-gen-test-1-npu-a3**: 作业在 git fetch 时，远程仓库返回 "not our ref 97caf460..."，说明该 commit 不存在或已被删除，可能是 PR 分支被强制更新或过期，属于环境/仓库状态问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175610

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查发现根因失败作业base-c-test-acc-8-npu-a3，触发Fast-fail机制，本作业未实际运行即被终止，属于依赖的其他作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175767

- **base-b-test-4-npu-a3 / run (0)**: 日志显示下载 actions/checkout@v4 时返回 429 Too Many Requests，重试后仍失败，属于 GitHub 服务端限流导致的环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175851

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业base-c-test-acc-8-npu-a3失败，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175930

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: codeload.github.com 返回 429 Too Many Requests，连续三次下载 actions/checkout 均失败，属于 GitHub 服务端限流导致的环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839176229

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: checkout 时 fetch 指定的 PR merge commit (97caf46) 失败，远端返回 'not our ref'，重试三次均失败，属于 Git 服务端或缓存同步问题，非代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839176331

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 actions/checkout 下载因 429 Too Many Requests 重试，随后 git fetch 多次失败，错误为 'not our ref 97caf460...'，可能是 PR 合并后 ref 失效或缓存问题，导致无法获取代码。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839176343

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 git fetch 时远程仓库返回 'not our ref 97caf460...'，多次重试均失败，说明该提交引用在远程仓库中不存在或已被删除，可能是 PR 已关闭或强制推送导致，属于环境/仓库状态问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839176369

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175755) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175777) |
| base-b-test-2-npu-a3 / run (0) | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175903) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32176324120/job/95839175959) |


## [Run #32176129512](https://github.com/sgl-project/sglang/actions/runs/32176129512)
- **分支**: `fix_ring_attention_npu`
- **总耗时**: 17.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32176129512

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.8min | 环境问题 | Git 拉取代码失败，远端仓库不存在指定 commit。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838518209) |
| multimodal-gen-test-2-npu-a3 | 4.0min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，且后续 git fetch 因 CDN 缓存未同步导致 ref 不存在。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838518439) |
| base-b-test-16-npu-a3 / run (0) | 1.2min | 环境问题 | 健康检查发现多个根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838518959) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519064) |
| base-b-test-4-npu-a3 / run (0) | 1.5min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519142) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519164) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 3.4min | 环境问题 | Git 拉取 PR 合并提交失败，远端仓库不存在该 ref。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519751) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | 环境问题 | GitHub Actions 下载 action 时遇到 429 限流，且 git fetch 时远程仓库缺少指定 ref，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519784) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.5min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，且后续 git fetch 因 PR 合并 ref 不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519838) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.7min | 环境问题 | Git 拉取 PR 合并提交失败，远端仓库缺少该 ref。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519859) |

- **multimodal-gen-test-1-npu-a3**: checkout 时 fetch 失败，报错 'not our ref f9f63a2e...'，重试三次均失败，可能是 PR 分支被删除或 commit 不存在，属于环境/仓库状态问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838518209

- **multimodal-gen-test-2-npu-a3**: 日志显示 actions/checkout 下载因 429 限流重试，随后 git fetch 时 CDN 返回 'not our ref'，多次重试均失败，最终作业因无法获取代码而退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838518439

- **base-b-test-16-npu-a3 / run (0)**: 日志显示base-c-test-acc系列作业（2/4/8/16）为根因失败，本作业因级联失败被过滤并快速失败，属于上游作业环境或代码问题引发的连锁失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838518959

- **base-b-test-2-npu-a3 / run (0)**: 本作业在健康检查阶段被快速失败机制跳过，根因是base-c-test-acc-8和acc-16两个作业失败，本作业并非实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519064

- **base-b-test-4-npu-a3 / run (0)**: 日志显示base-c-test-acc-8和acc-16两个作业失败被判定为根因，健康检查脚本据此触发fast-fail，导致本作业未实际运行即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519142

- **base-b-test-1-npu-a3 / run (0)**: 该作业在健康检查阶段检测到根因作业base-c-test-acc-8/16-npu-a3失败，触发fast-fail机制，本作业被级联取消，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519164

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: checkout 时尝试获取 PR #34855 的合并提交 f9f63a2，但远端 git-cdn 服务返回 'not our ref'，多次重试均失败，导致作业无法开始。可能是 PR 已关闭或缓存未同步。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519751

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 actions/checkout 和 github-script 下载均因 429 Too Many Requests 失败重试；随后 git fetch 指定 commit f9f63a2 时远程返回 'not our ref'，多次重试仍失败，最终退出码 1。属于基础设施或仓库状态问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519784

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在准备阶段下载 actions/checkout 时遭遇 GitHub 429 Too Many Requests 限流，重试后虽成功，但后续 git fetch 指定 PR 合并 ref（f9f63a2...）时，远端仓库返回 'not our ref'，导致多次重试均失败，最终作业退出。这属于基础设施或远端仓库状态问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519838

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: checkout 时 fetch 指定 commit f9f63a2 失败，报错 'not our ref'，重试三次均失败，可能是 PR 已更新或缓存未同步，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519859

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838518770) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838518955) |
| base-b-test-8-npu-a3 / run (0) | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32176129512/job/95838519154) |


## [Run #32175797035](https://github.com/sgl-project/sglang/actions/runs/32175797035)
- **分支**: `codex/vae-tiling-and-nvfp4-gate`
- **总耗时**: 6.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32175797035

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.3min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175797035/job/95837439515) |

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法看到具体测试命令和错误输出。仅能确认上传diffusion-failures目录时未找到文件，说明测试可能未产生失败产物，但实际失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175797035/job/95837439515


## [Run #32175384720](https://github.com/sgl-project/sglang/actions/runs/32175384720)
- **分支**: `mick/dit-tp-shard-planner`
- **总耗时**: 8.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32175384720

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.8min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175384720/job/95836173155) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能确认上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业最终失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175384720/job/95836173155


## [Run #32175172332](https://github.com/sgl-project/sglang/actions/runs/32175172332)
- **分支**: `codex/diffusion-use-checkpoint-quant-spec`
- **总耗时**: 12.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32175172332

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425298) |
| base-b-test-4-npu-a3 / run (0) | 2.2min | 环境问题 | GitHub Actions 下载 actions/checkout 时遭遇 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425353) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425377) |
| base-b-test-2-npu-a3 / run (0) | 1.4min | 其他 | 健康检查发现其他作业根因失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425396) |
| base-b-test-8-npu-a3 / run (0) | 1.7min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425511) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425518) |
| multimodal-gen-test-1-npu-a3 | 5.5min | 环境问题 | GitHub Actions 下载 actions/checkout 时遭遇 429 限流，导致重试延迟。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425534) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835426301) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835426375) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.0min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835426391) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835426448) |

- **base-b-test-1-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3为根因失败作业，本作业因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425298

- **base-b-test-4-npu-a3 / run (0)**: 日志显示下载 actions/checkout@v4 时多次返回 429 Too Many Requests，重试后仍失败，最终触发健康检查快速失败机制，导致本作业被标记为失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425353

- **base-b-test-4-npu-a3 / run (1)**: 健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425377

- **base-b-test-2-npu-a3 / run (0)**: 本作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而快速失败，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425396

- **base-b-test-8-npu-a3 / run (0)**: 日志显示下载 actions/checkout@v4 时返回 429 Too Many Requests，重试后仍失败，属于 GitHub 服务端限流导致的环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425511

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业因快速失败策略被终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425518

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/checkout 时返回 429 Too Many Requests，重试两次后成功，但增加了约 1 分钟延迟。作业最终正常完成，无测试失败迹象，属于临时网络限流问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425534

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835426301

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业在启动前被健康检查拦截，原因是根因作业multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835426375

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835426391

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835426448

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32175172332/job/95835425485) |


## [Run #32175114286](https://github.com/sgl-project/sglang/actions/runs/32175114286)
- **分支**: `codex/diffusion-modelopt-exact-dispatch`
- **总耗时**: 9.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32175114286

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.3min | 其他 | 日志不完整，未显示测试执行过程，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835263604) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835263890) |
| base-b-test-4-npu-a3 / run (0) | 3.9min | 环境问题 | GitHub Actions 下载 actions/checkout 时遭遇 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835263943) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查快速失败，多个作业被标记为根因失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835263944) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | 环境问题 | 测试文件执行失败，NPU后端出现未知异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264025) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264131) |
| base-b-test-16-npu-a3 / run (0) | 3.7min | 环境问题 | GitHub Actions 下载 action 时遇到 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264188) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264240) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | PR测试健康检查失败，导致作业被快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264884) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264907) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查失败，因其他根因作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264915) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835265049) |

- **multimodal-gen-test-1-npu-a3**: 日志被截断，中间省略了关键测试步骤。仅能看到上传diffusion-failures工件时提示无文件，说明测试可能未产生失败记录，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835263604

- **base-b-test-1-npu-a3 / run (0)**: 本作业在启动前的健康检查中检测到其他三个根因作业（multimodal-gen-test-1-npu-a3、base-b-test-4-npu-a3、base-a-test-1-npu-a2）已失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835263890

- **base-b-test-4-npu-a3 / run (0)**: 日志显示下载 actions/checkout@v4 时返回 429 Too Many Requests，重试后仍失败，最终导致作业终止。属于 GitHub 服务端限流问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835263943

- **base-b-test-4-npu-a3 / run (1)**: 作业因健康检查检测到多个根因失败（如multimodal-gen-test-1-npu-a3等）而触发快速失败机制，并非本作业自身问题，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835263944

- **base-a-test-1-npu-a2 / run (0)**: test_npu_ascend_backend.py 在NPU上运行时报 ERR99999 UNKNOWN application exception，导致测试失败退出码1，可能是NPU环境或驱动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264025

- **base-b-test-2-npu-a3 / run (0)**: 作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3、base-b-test-4-npu-a3和base-a-test-1-npu-a2三个根因作业失败，因此触发fast-fail跳过本作业，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264131

- **base-b-test-16-npu-a3 / run (0)**: 下载 actions/github-script@v8 时返回 429 Too Many Requests，重试后仍失败，属于 GitHub 服务端限流问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264188

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3、base-b-test-4-npu-a3和base-a-test-1-npu-a2等根因作业失败，本作业作为级联失败被过滤，最终因快速失败策略退出，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264240

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业因PR测试健康检查失败被快速失败机制跳过，根因是其他作业（multimodal-gen-test-1-npu-a3等）失败，本作业并非实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264884

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到多个根因失败作业（如multimodal-gen-test-1-npu-a3等），触发fast-fail机制，本作业在启动前被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264907

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 该作业在启动前被健康检查过滤，根因是multimodal-gen-test-1-npu-a3、base-b-test-4-npu-a3和base-a-test-1-npu-a2等作业失败，触发fast-fail机制，本作业未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835264915

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查发现根因失败作业（multimodal-gen-test-1-npu-a3等），触发fast-fail机制，本作业未实际运行即被终止，属于级联跳过而非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175114286/job/95835265049


## [Run #32175052594](https://github.com/sgl-project/sglang/actions/runs/32175052594)
- **分支**: `codex/vae-quantized-repo-routing`
- **总耗时**: 31.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32175052594

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.8min | 环境问题 | GitHub Actions 下载 action 时遭遇 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112550) |
| base-b-test-8-npu-a3 / run (0) | 1.4min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112759) |
| base-b-test-16-npu-a3 / run (0) | 1.2min | 其他 | 健康检查快速失败，因其他根因作业失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112816) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112934) |
| base-b-test-4-npu-a3 / run (1) | 1.4min | 环境问题 | GitHub Actions 下载 checkout 动作时遇到 429 限流，重试后仍失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112997) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835113176) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.7min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835113331) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.2min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835113402) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查检测到根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835113424) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/checkout 和 upload-artifact 时多次返回 429 Too Many Requests，重试后仍失败，属于 GitHub 服务端限流导致的环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112550

- **base-b-test-8-npu-a3 / run (0)**: 日志显示下载 actions/checkout@v4 时返回 429 Too Many Requests，重试后仍失败，属于 GitHub 服务端限流问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112759

- **base-b-test-16-npu-a3 / run (0)**: 本作业未实际运行测试，在健康检查阶段检测到根因作业base-b-test-4-npu-a3失败，触发fast-fail机制跳过，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112816

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现base-b-test-4-npu-a3作业失败，触发快速失败机制，本作业未实际运行即被终止，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112934

- **base-b-test-4-npu-a3 / run (1)**: 作业在准备阶段下载 actions/checkout@v4 时，codeload.github.com 返回 429 Too Many Requests，重试 3 次后仍失败，导致作业无法启动。属于 GitHub 服务端限流或网络问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112997

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-4-npu-a3/run，触发fast-fail机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835113176

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到base-b-test-4-npu-a3作业失败，将其判定为根因失败，因此本作业被快速失败跳过，并非自身代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835113331

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业在健康检查阶段因检测到根因作业 base-b-test-4-npu-a3 / run (1) 失败而触发 fast-fail，本作业被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835113402

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是base-b-test-4-npu-a3 / run (1)失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835113424

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112670) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112873) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32175052594/job/95835112884) |


## [Run #32174861977](https://github.com/sgl-project/sglang/actions/runs/32174861977)
- **分支**: `mick/diffusion-auto-residency`
- **总耗时**: 7.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32174861977

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32174861977/job/95834431073) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact（未找到文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32174861977/job/95834431073


## [Run #32174081582](https://github.com/sgl-project/sglang/actions/runs/32174081582)
- **分支**: `pd/abort-on-waiting-timeout`
- **总耗时**: 109.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32174081582

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 环境问题 | GitHub Actions 下载 actions/checkout 时遭遇 429 限流，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920547) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 73.4min | 环境问题 | NPU服务器启动失败导致测试未执行 | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95840726133) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 14.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95859449805) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/checkout@v4 时返回 429 Too Many Requests，触发重试后仍未能成功获取，属于 GitHub 服务端限流导致的环境问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920547

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: sglang serve命令启动服务器时进程退出（code 1），测试集未运行即报错。日志显示服务器启动失败，可能是模型加载、配置或NPU环境问题，需检查服务器日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95840726133

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示Prefill/Decode正常进行，但突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95859449805

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920500) |
| base-b-test-16-npu-a3 / run (0) | 50.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920532) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920675) |
| base-b-test-8-npu-a3 / run (0) | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920713) |
| base-b-test-4-npu-a3 / run (0) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920718) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920737) |
| base-a-test-1-npu-a2 / run (0) | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832920869) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832921069) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832921458) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832921470) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95832921706) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95835390288) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32174081582/job/95845193790) |


## [Run #32173758760](https://github.com/sgl-project/sglang/actions/runs/32173758760)
- **分支**: `main`
- **总耗时**: 52.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32173758760

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.7min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905526) |
| base-b-test-1-npu-a3 / run (0) | 33.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905689) |
| base-b-test-4-npu-a3 / run (0) | 7.8min | 代码错误 | HiCache MLA测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905763) |
| base-b-test-16-npu-a3 / run (0) | 32.9min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905787) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 33.7min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833906044) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 29.1min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833906189) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.8min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 429 限流，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95844318762) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业正常执行，上传diffusion-failures目录时提示无文件，未发现测试失败或错误信息，可能为测试通过但无失败产物，或日志截断导致关键错误未显示。
  链接: https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905526

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行到49%时，出现'Executing the custom container implementation failed'错误，属于runner容器环境问题，而非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905689

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试执行失败（exit code 1），耗时271秒，导致整个作业失败。具体失败原因需查看该测试的详细输出，可能是功能实现或测试用例本身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905763

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动NPU测试容器时失败，日志显示所有TP/EP进程获取ASCEND_OPP_PATH后，容器执行报错，可能是CANN环境配置或容器资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905787

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（有吞吐数据），但突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833906044

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行约30分钟后，在Decode阶段正常输出时突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833906189

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示下载 actions/checkout@v4 时返回 429 Too Many Requests，重试后仍失败，最终报错“Executing the custom container implementation failed”，属于 GitHub 服务端限流导致的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95844318762

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905581) |
| base-b-test-2-npu-a3 / run (0) | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905671) |
| base-b-test-8-npu-a3 / run (0) | 11.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905799) |
| base-b-test-4-npu-a3 / run (1) | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833905896) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833906236) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95833906251) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32173758760/job/95835576895) |


## [Run #32171106386](https://github.com/sgl-project/sglang/actions/runs/32171106386)
- **分支**: `main`
- **总耗时**: 33.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32171106386

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 环境问题 | 作业因缺少失败产物文件而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694289) |
| base-b-test-4-npu-a3 / run (0) | 9.0min | 超时 | NPU测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694631) |
| base-b-test-16-npu-a3 / run (0) | 23.5min | 环境问题 | NPU容器执行失败，模型权重加载时发生内存错误导致进程崩溃 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694718) |
| base-b-test-4-npu-a3 / run (1) | 24.6min | 环境问题 | 自定义容器执行失败，NPU分布式初始化未完成 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694735) |
| base-b-test-2-npu-a3 / run (0) | 27.1min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694845) |
| base-b-test-1-npu-a3 / run (0) | 27.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694917) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.1min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822695339) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822695547) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 10.5min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822695675) |

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures目录时提示无文件，说明测试未产生失败样本，作业可能因前置条件未满足或测试未运行而终止，属于环境或流程配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694289

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试运行301秒后超时（估计时间400秒），返回退出码1，导致整个作业失败。测试用例可能因NPU资源问题或代码性能问题未能及时完成。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694631

- **base-b-test-16-npu-a3 / run (0)**: 在加载MoE模型权重时，copy_操作触发底层内存错误，随后Scheduler watchdog超时，最终自定义容器执行失败。可能是NPU显存不足或驱动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694718

- **base-b-test-4-npu-a3 / run (1)**: 日志显示torch分布式初始化开始后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694735

- **base-b-test-2-npu-a3 / run (0)**: 测试运行到第5个文件时，自定义容器实现执行失败，导致作业提前终止。日志显示测试本身通过（OK），但容器环境问题导致后续任务无法继续。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694845

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694917

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常（吞吐量正常），但在18:56:35时出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822695339

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常（吞吐量正常），但突然报错“Executing the custom container implementation failed”，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822695547

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装依赖后，执行自定义容器时出现错误，提示联系自托管runner管理员，可能是容器环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822695675

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822444182) |
| base-b-test-8-npu-a3 / run (0) | 11.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694756) |
| base-a-test-1-npu-a2 / run (0) | 8.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822694846) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95822695768) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32171106386/job/95825268337) |


## [Run #32168566803](https://github.com/sgl-project/sglang/actions/runs/32168566803)
- **分支**: `perf/prefill-nonblocking-h2d`
- **总耗时**: 330.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32168566803

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814548515) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 130.8min | 精度回归 | qwen3_5_9b 精度测试失败，导致作业整体失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814549671) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 303.5min | 超时 | 性能测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95821784523) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions的常规准备、上传artifact（无文件）和清理步骤。无法判断测试是否失败或失败原因，可能日志被截断或作业在测试前已中止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814548515

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件中 qwen3_5_9b_bf16_1p_gsm8k 测试退出码为1，而其他两个测试通过，表明该模型存在精度回归问题，需检查模型输出或配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814549671

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试 test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py 在启动服务器后运行超过7800秒未完成，被强制终止，最终0/4测试通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95821784523

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 50.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814548602) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814548628) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814548654) |
| base-b-test-1-npu-a3 / run (0) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814548663) |
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814548722) |
| base-b-test-4-npu-a3 / run (0) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814548935) |
| base-b-test-8-npu-a3 / run (0) | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814549057) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814549326) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814549367) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95814549712) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95815944721) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32168566803/job/95826382651) |


## [Run #32167618314](https://github.com/sgl-project/sglang/actions/runs/32167618314)
- **分支**: `feat/nixl-deferred-decode-kv-release`
- **总耗时**: 109.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32167618314

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.3min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818631561) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 46.7min | 性能回归 | deepseek_v4_flash性能测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95829719297) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因同批次其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95850982457) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传diffusion-failures工件（未找到文件）的信息，未出现任何测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818631561

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py测试失败（exit code 1），耗时仅334秒，远低于其他测试，疑似性能未达标或运行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95829719297

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-16-npu-a3，本作业被快速失败机制跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95850982457

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818631684) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818631692) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818631772) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818631812) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818631955) |
| base-b-test-2-npu-a3 / run (0) | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818632016) |
| base-b-test-16-npu-a3 / run (0) | 55.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818632183) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818632519) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818632605) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818632608) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95818632681) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95820260984) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32167618314/job/95832024817) |


## [Run #32166937173](https://github.com/sgl-project/sglang/actions/runs/32166937173)
- **分支**: `mmangkad/pd-decode-retraction-host-pool-capacity-fallback`
- **总耗时**: 352.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32166937173

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808811416) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 305.4min | 超时 | 性能测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95820869656) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 28.2min | 性能回归 | NPU性能测试中qwen3_6_27b_w8a8_1p_in64k_out1k_50ms用例失败，疑似性能未达标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95834266333) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅显示Node.js 20弃用警告和上传diffusion-failures目录时未找到文件。可能测试未运行或日志被截断，需查看完整日志确认失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808811416

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试 test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py 在启动服务器后运行超过7800秒未完成，被强制终止，最终0/4测试通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95820869656

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试套件中两个用例通过，但test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py退出码为1，属于性能回归，可能因长序列场景下吞吐或延迟未达阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95834266333

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808811692) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808811835) |
| base-b-test-2-npu-a3 / run (0) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808811854) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808811992) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808812031) |
| base-b-test-16-npu-a3 / run (0) | 50.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808812042) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808812090) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808812440) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808812456) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808812526) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95808812531) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95812291320) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166937173/job/95820904017) |


## [Run #32166635316](https://github.com/sgl-project/sglang/actions/runs/32166635316)
- **分支**: `dev-dsv4-gb300-tlru`
- **总耗时**: 349.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32166635316

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.7min | 其他 | 作业日志被截断，未显示实际测试结果，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807777488) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 304.6min | 超时 | 性能测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95819496577) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 28.7min | 性能回归 | NPU性能测试中qwen3_6_27b_w8a8长序列用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95834811831) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业整体状态未知，需查看完整日志判断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807777488

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py 测试超过7800秒超时限制，最终测试结果1/4通过，该用例未在限定时间内完成，可能因模型推理性能不足或环境负载过高导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95819496577

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py退出码1，该用例为64k长输入性能测试，可能因性能未达阈值或资源限制导致失败，其余两个用例通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95834811831

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807779166) |
| base-b-test-1-npu-a3 / run (0) | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807779214) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807779268) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807779301) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807779364) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807779457) |
| base-b-test-16-npu-a3 / run (0) | 54.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807779486) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807781498) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807781528) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807781552) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95807781710) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95810154622) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166635316/job/95819659176) |


## [Run #32166009777](https://github.com/sgl-project/sglang/actions/runs/32166009777)
- **分支**: `RM/fix-dsa-paged-tree-relocation`
- **总耗时**: 333.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32166009777

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763195) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 303.6min | 超时 | NPU性能测试用例超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95813139190) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 33.2min | 性能回归 | NPU性能测试中qwen3_6_27b_w8a8用例失败，未达到50ms性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95831577463) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败产物，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763195

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py测试超过7800秒超时，4个测试中仅1个通过，其余因超时失败，属于性能测试超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95813139190

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试套件中4个性能用例有3个通过，但test_npu_qwen3_6_27b_w8a8_1p_in64k_out1k_50ms.py退出码1，耗时669秒，未满足性能要求，疑似该模型配置下性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95831577463

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 54.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763377) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763499) |
| base-b-test-8-npu-a3 / run (0) | 10.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763535) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763543) |
| base-b-test-4-npu-a3 / run (0) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763582) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763626) |
| base-b-test-2-npu-a3 / run (0) | 19.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805763691) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805764662) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805764663) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805764766) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95805764782) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95806986536) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32166009777/job/95817231115) |


## [Run #32165942156](https://github.com/sgl-project/sglang/actions/runs/32165942156)
- **分支**: `main`
- **总耗时**: 58.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32165942156

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 8.0min | 环境问题 | NPU测试用例test_npu_hicache_mla.py执行失败，返回退出码1，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180666) |
| multimodal-gen-test-1-npu-a3 | 3.8min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180717) |
| base-b-test-16-npu-a3 / run (0) | 54.7min | 环境问题 | 自定义容器执行失败，NPU测试中途终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180851) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 54.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806181132) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95808357166) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95814709283) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95817676028) |

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test/registered/npu/basic_function/HiCache/test_npu_hicache_mla.py在运行281秒后失败，退出码为1，最终作业因非零退出码255终止。可能是NPU环境配置或测试用例本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180666

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体信息，仅有GitHub Actions环境准备、Node版本警告及上传artifact（无文件）等常规步骤，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180717

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试在捕获批次时可用内存持续下降（9.62GB降至9.43GB），随后报错“Executing the custom container implementation failed”，属于自托管runner容器环境异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180851

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试请求均返回200 OK，但随后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806181132

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动阶段被健康检查拦截，检测到同一运行中已有根因失败作业（base-b-test-4-npu-a3 / run），触发fast-fail机制，本作业未实际执行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95808357166

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到根因作业base-b-test-4-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即被取消，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95814709283

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段被健康检查拦截，检测到根因作业base-b-test-4-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95817676028

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 1.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95805799186) |
| base-b-test-1-npu-a3 / run (0) | 46.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180694) |
| base-b-test-4-npu-a3 / run (1) | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180697) |
| base-b-test-8-npu-a3 / run (0) | 12.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180698) |
| base-b-test-2-npu-a3 / run (0) | 28.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180750) |
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806180853) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806181169) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806181193) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32165942156/job/95806181231) |


## [Run #32165069073](https://github.com/sgl-project/sglang/actions/runs/32165069073)
- **分支**: `main`
- **总耗时**: 11.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32165069073

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取必要数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740460) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740612) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | 其他 | 作业实际成功，日志显示所有测试通过，无失败迹象。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740640) |
| base-b-test-1-npu-a3 / run (0) | 7.0min | 环境问题 | 测试本身通过，但自定义容器执行失败导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740642) |
| base-b-test-4-npu-a3 / run (1) | 6.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740653) |
| base-b-test-4-npu-a3 / run (0) | 7.0min | 环境问题 | 自定义容器执行失败，NPU测试中途中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740654) |
| base-b-test-16-npu-a3 / run (0) | 5.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740678) |
| base-b-test-2-npu-a3 / run (0) | 7.0min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740737) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803741057) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.9min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803741105) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803741106) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.0min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803741131) |

- **multimodal-gen-test-1-npu-a3**: 作业在下载或访问日志文件时，Azure Blob 返回 BlobNotFound 错误，说明该 blob 已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740460

- **base-b-test-8-npu-a3 / run (0)**: 日志显示测试运行中容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740612

- **base-a-test-1-npu-a2 / run (0)**: 日志显示42个测试全部通过，2个测试文件均PASSED，作业正常结束。仅有Node.js 20弃用警告，但未影响执行。可能为误报或作业状态标记异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740640

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行成功（OK，Accuracy 0.868），但随后出现错误：Executing the custom container implementation failed，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740642

- **base-b-test-4-npu-a3 / run (1)**: 日志显示Prefill批处理正常进行，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740653

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试在捕获批次时（bs=4）突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740654

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型权重加载到25%时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU测试环境或容器配置问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740678

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动NPU推理引擎时，TokenizerManager初始化后立即报错"Executing the custom container implementation failed"，表明自托管runner的容器环境存在问题，导致测试无法继续执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803740737

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803741057

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在安装evalscope依赖构建wheel时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803741105

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803741106

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载模型分片（约77%）时，自定义容器实现执行失败，导致CI中断。可能是容器环境不稳定或资源限制，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32165069073/job/95803741131


## [Run #32163916599](https://github.com/sgl-project/sglang/actions/runs/32163916599)
- **分支**: `main`
- **总耗时**: 16.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32163916599

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799017996) |
| base-b-test-16-npu-a3 / run (0) | 12.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018025) |
| multimodal-gen-test-1-npu-a3 | 3.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018038) |
| base-b-test-4-npu-a3 / run (0) | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018080) |
| base-b-test-1-npu-a3 / run (0) | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018106) |
| base-b-test-4-npu-a3 / run (1) | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018107) |
| base-b-test-8-npu-a3 / run (0) | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018122) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018548) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018553) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018566) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | 其他 | 测试套件未找到任何测试用例，属于配置或迁移问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018591) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799017996

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018025

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤（无文件上传），未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018038

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018080

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018106

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018107

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018122

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径错误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018548

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018553

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018566

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示“No tests found for hw=NPU, suite=base-c-test-acc-8-npu-a3”，测试直接跳过，未执行任何实际测试。后续脚本因测试计数为0出现整数表达式错误，但非根本原因。可能是测试用例未注册或套件名称不匹配。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018591

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163916599/job/95799018176) |


## [Run #32163799367](https://github.com/sgl-project/sglang/actions/runs/32163799367)
- **分支**: `codex/minimax-h3-cube-sparse-attn`
- **总耗时**: 6.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32163799367

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163799367/job/95798683233) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163799367/job/95798683233


## [Run #32163650848](https://github.com/sgl-project/sglang/actions/runs/32163650848)
- **分支**: `mick/diffusion-auto-residency`
- **总耗时**: 8.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32163650848

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163650848/job/95798217826) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node.js版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163650848/job/95798217826


## [Run #32163536232](https://github.com/sgl-project/sglang/actions/runs/32163536232)
- **分支**: `user/yawei_microsoft/fix-nixl-hybrid-cleaner-grouping`
- **总耗时**: 114.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32163536232

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866279) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.3min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797867334) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95803226365) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 79.0min | 环境问题 | NPU服务器启动失败，导致测试未执行 | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95807158672) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。实际失败原因可能被截断或未记录，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866279

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行约1.5小时后，在Decode阶段正常输出时突然报错"Executing the custom container implementation failed"，属于自托管runner容器环境问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797867334

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1154秒后失败，该测试为性能测试，预期50ms延迟，实际未达标，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95803226365

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: sglang serve命令启动DeepSeek-V4-Flash模型时服务器进程退出（code 1），测试在setUpClass阶段失败，0个测试被执行。可能是模型加载、NPU资源或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95807158672

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866435) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866436) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866456) |
| base-b-test-4-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866460) |
| base-b-test-16-npu-a3 / run (0) | 44.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866486) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866538) |
| base-b-test-2-npu-a3 / run (0) | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797866558) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797867188) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797867217) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95797867283) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32163536232/job/95812964019) |


## [Run #32160349439](https://github.com/sgl-project/sglang/actions/runs/32160349439)
- **分支**: `claude/paddleocr-support-optimization-bfaf52`
- **总耗时**: 96.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32160349439

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.1min | 其他 | 日志不完整，未显示测试执行过程，仅见上传失败产物时无文件，作业最终状态不明。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127416) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 19.6min | 性能回归 | NPU性能测试中deepseek_v4_flash用例失败，未达性能目标 | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95799762371) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，因同一PR中另一个性能测试作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95802033980) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 该作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95813848272) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions基础设施信息（Node 20弃用警告、上传artifact无文件），未展示实际测试命令和结果，无法判断失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127416

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中qwen3_5_397b通过，但deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms用例退出码1，耗时283秒，未通过性能验证，可能因模型性能未达预期或环境波动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95799762371

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业未实际运行，在健康检查阶段检测到同PR的base-c-test-perf-16-npu-a3作业失败，被判定为根因失败，触发fast-fail跳过本作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95802033980

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示根因失败作业为base-c-test-perf-16-npu-a3，本作业作为级联失败被过滤后触发fast-fail，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95813848272

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127555) |
| base-b-test-16-npu-a3 / run (0) | 51.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127654) |
| base-b-test-2-npu-a3 / run (0) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127667) |
| base-b-test-8-npu-a3 / run (0) | 8.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127689) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127745) |
| base-a-test-1-npu-a2 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127890) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789127919) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789128000) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789128020) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789128077) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95789128120) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160349439/job/95791310904) |


## [Run #32160109975](https://github.com/sgl-project/sglang/actions/runs/32160109975)
- **分支**: `claude/dynamic-per-request-config-20760a`
- **总耗时**: 8.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32160109975

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32160109975/job/95786965535) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions的初始化、Node版本警告和上传artifact步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32160109975/job/95786965535

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32160109975/job/95786710257) |


## [Run #32159380581](https://github.com/sgl-project/sglang/actions/runs/32159380581)
- **分支**: `main`
- **总耗时**: 17.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32159380581

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32159380581/job/95784688404) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32159380581/job/95784688404


## [Run #32158717122](https://github.com/sgl-project/sglang/actions/runs/32158717122)
- **分支**: `codex/vae-tiling-and-nvfp4-gate`
- **总耗时**: 8.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32158717122

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 环境问题 | Git 拉取代码失败，远端仓库缺少指定 commit | [job link](https://github.com/sgl-project/sglang/actions/runs/32158717122/job/95797395923) |

- **multimodal-gen-test-1-npu-a3**: checkout 时 fetch PR merge commit d818bd26 失败，远端报 'not our ref'，重试三次均失败，属于仓库状态或缓存不一致导致的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158717122/job/95797395923


## [Run #32158668938](https://github.com/sgl-project/sglang/actions/runs/32158668938)
- **分支**: `codex/comfyui-minimax-h3-node`
- **总耗时**: 9.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32158668938

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158668938/job/95797406801) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示upload-artifact步骤因diffusion-failures目录无文件而跳过，可能测试未产生失败样本或测试未运行。需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158668938/job/95797406801

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32158668938/job/95797372236) |


## [Run #32158423544](https://github.com/sgl-project/sglang/actions/runs/32158423544)
- **分支**: `main`
- **总耗时**: 10.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32158423544

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615155) |
| base-b-test-1-npu-a3 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，NPU测试在加载模型权重时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615231) |
| base-b-test-4-npu-a3 / run (1) | 5.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615291) |
| base-b-test-4-npu-a3 / run (0) | 5.3min | 环境问题 | NPU图捕获过程中容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615305) |
| base-b-test-2-npu-a3 / run (0) | 4.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615315) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615419) |
| base-b-test-16-npu-a3 / run (0) | 8.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615428) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615840) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.5min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615938) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615999) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.4min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781616006) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的模型或数据文件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615155

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在加载模型权重分片（25%）时，GitHub Actions报错“Executing the custom container implementation failed”，属于自托管runner容器环境异常，非代码或测试逻辑问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615231

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在加载模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615291

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在捕获decode NPU图时（bs=256）出现"Executing the custom container implementation failed"错误，可能是NPU显存不足或容器环境异常，导致自托管runner无法继续执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615305

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在加载模型权重时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615315

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615419

- **base-b-test-16-npu-a3 / run (0)**: 作业失败原因是Azure Blob存储返回BlobNotFound错误，表明CI流程尝试下载或访问的blob文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615428

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615840

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在加载模型权重时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615938

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615999

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在模型权重加载阶段（Multi-thread loading shards 0%）时，GitHub Actions 报错“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于容器环境或 runner 配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781616006

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32158423544/job/95781615461) |


## [Run #32157490647](https://github.com/sgl-project/sglang/actions/runs/32157490647)
- **分支**: `main`
- **总耗时**: 10.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32157490647

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626085) |
| base-b-test-4-npu-a3 / run (0) | 7.9min | 环境问题 | 自定义容器执行失败，NPU后端算子不支持导致回退 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626396) |
| base-b-test-1-npu-a3 / run (0) | 7.8min | 环境问题 | NPU图捕获阶段容器执行失败，自托管runner异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626504) |
| base-b-test-4-npu-a3 / run (1) | 7.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626522) |
| base-b-test-2-npu-a3 / run (0) | 7.8min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626548) |
| base-b-test-8-npu-a3 / run (0) | 7.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626599) |
| base-b-test-16-npu-a3 / run (0) | 5.3min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626605) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.0min | 环境问题 | 自定义容器执行失败，模型权重加载过程中容器异常退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778627168) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778627303) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778627387) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778627522) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未显示multimodal测试执行过程或失败断言，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626085

- **base-b-test-4-npu-a3 / run (0)**: 日志显示aten::_assert_async算子不支持NPU后端，回退到CPU执行，随后自定义容器实现执行失败，提示联系自托管runner管理员，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626396

- **base-b-test-1-npu-a3 / run (0)**: 在decode NPU graph捕获过程中（bs=64时），自定义容器实现执行失败，提示联系runner管理员，属于NPU环境或容器运行时问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626504

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626522

- **base-b-test-2-npu-a3 / run (0)**: 测试在运行ExpertDistributionRecorderModeStatic测试时，自定义容器实现执行失败，导致作业提前终止。日志显示容器环境存在问题，而非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626548

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626599

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型权重加载到31%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626605

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在加载模型权重（约11%进度）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境或容器问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778627168

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778627303

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778627387

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778627522

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32157490647/job/95778626758) |


## [Run #32157335285](https://github.com/sgl-project/sglang/actions/runs/32157335285)
- **分支**: `claude/paddleocr-support-optimization-bfaf52`
- **总耗时**: 37.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32157335285

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112378) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112561) |
| base-b-test-16-npu-a3 / run (0) | 27.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112574) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | 其他 | 作业实际成功，日志显示测试全部通过，无失败迹象。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112959) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 29.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778113003) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778113121) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 11.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95779902366) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤（无文件上传），未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112378

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载Meta-Llama-3-8B-Instruct模型时，torch_npu的transfer_to_npu模块发出警告，随后自定义容器实现执行失败，导致作业终止。可能是NPU驱动或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112561

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型分片加载到89%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112574

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试用例 TestNPUGLM5_Top64_Pruned_GSM8K 通过，准确率0.48与基线一致，测试总结为1/1 passed，作业正常结束，无错误或失败信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112959

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行正常，但在执行过程中出现错误："Executing the custom container implementation failed. Please contact your self hosted runner administrator."，表明自托管运行器环境问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778113003

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在加载模型分片时（约92%）自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778113121

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试运行正常，但在Prefill阶段后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95779902366

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112436) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112514) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112525) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112568) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112619) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32157335285/job/95778112908) |


## [Run #32156721326](https://github.com/sgl-project/sglang/actions/runs/32156721326)
- **分支**: `main`
- **总耗时**: 9.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32156721326

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045694) |
| base-b-test-2-npu-a3 / run (0) | 6.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045743) |
| base-b-test-16-npu-a3 / run (0) | 6.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045764) |
| base-b-test-8-npu-a3 / run (0) | 5.3min | 环境问题 | GitHub Actions 下载 actions/checkout 时遇到 503 服务不可用，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045785) |
| base-b-test-4-npu-a3 / run (0) | 6.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045867) |
| base-b-test-4-npu-a3 / run (1) | 6.2min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045873) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045906) |
| base-b-test-1-npu-a3 / run (0) | 6.1min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045908) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.0min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045951) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776046104) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.2min | 环境问题 | 自定义容器执行失败，测试未真正运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776046154) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95777601932) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045694

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试本身通过（HTTP 200），但随后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045743

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045764

- **base-b-test-8-npu-a3 / run (0)**: 日志显示下载 actions/checkout@v4 时返回 503 Service Unavailable，重试后仍失败，最终自定义容器执行失败，属于 GitHub 服务端临时故障或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045785

- **base-b-test-4-npu-a3 / run (0)**: 日志显示在捕获批次过程中，自定义容器实现执行失败，提示联系自托管运行器管理员，属于运行环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045867

- **base-b-test-4-npu-a3 / run (1)**: 作业在加载模型权重后，执行自定义容器实现时失败，提示联系自托管runner管理员，属于NPU容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045873

- **base-a-test-1-npu-a2 / run (0)**: 第二个测试test_npu_ascend_dsv4_backend.py启动后，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU运行环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045906

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045908

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装evalscope等依赖后，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776045951

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试服务已成功启动并完成一次请求，但随后出现错误："Executing the custom container implementation failed"，表明自托管runner在运行自定义容器时遇到环境问题，导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776046104

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示测试命令已启动，但随后报错“Executing the custom container implementation failed”，说明自托管runner的容器环境异常，导致测试进程未能正常执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776046154

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95777601932

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32156721326/job/95776046204) |


## [Run #32154977313](https://github.com/sgl-project/sglang/actions/runs/32154977313)
- **分支**: `tvm_ffi`
- **总耗时**: 76.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32154977313

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052071) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查失败：lint检查未通过导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052643) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052658) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 71.2min | 精度回归 | NPU精度测试用例失败，0/3通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770053326) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.3min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95771459114) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | PR健康检查中的lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95777773495) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查失败：lint检查未通过，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95783444613) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含multimodal-gen测试的具体执行步骤或错误信息，仅有GitHub Actions基础设施警告（Node 20弃用）和上传artifact时无文件提示，无法判断测试失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052071

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052643

- **base-b-test-1-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因作业，因此本作业（base-b-test-1-npu-a3）被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052658

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py执行4069秒后失败，返回码1，所有3个测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770053326

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1057秒后退出码为1，0/1通过，属于性能指标未达预期。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95771459114

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查结论为failure，触发了fast-fail机制，作业在真正运行测试前即被终止，属于前置检查拦截而非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95777773495

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业立即失败，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95783444613

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 52.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052460) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052532) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052600) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052639) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770052665) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770053251) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770053252) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154977313/job/95770053364) |


## [Run #32154625018](https://github.com/sgl-project/sglang/actions/runs/32154625018)
- **分支**: `claude/mm-processor-concurrency-default`
- **总耗时**: 59.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32154625018

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 其他 | 作业未显示明确失败原因，日志仅包含正常执行和Node 20弃用警告。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797355630) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797355906) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797355960) |
| base-b-test-16-npu-a3 / run (0) | 3.0min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356079) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356823) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356971) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356974) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.0min | 其他 | 健康检查失败，根因是多模态测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95808410664) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业正常执行，上传artifact时未找到diffusion-failures文件（if-no-files-found: ignore），最终正常清理退出。无测试失败、超时或错误信息，可能为作业被取消或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797355630

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段因健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797355906

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-1-npu-a3）被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797355960

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356079

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356823

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤掉级联失败后，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356971

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤掉级联失败后，根因作业为multimodal-gen-test-1-npu-a3，本作业因Fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356974

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，本作业被标记为级联失败并快速跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95808410664

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797355930) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356094) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356131) |
| base-b-test-4-npu-a3 / run (0) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356174) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 34.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154625018/job/95797356738) |


## [Run #32154231117](https://github.com/sgl-project/sglang/actions/runs/32154231117)
- **分支**: `claude/paddleocr-support-optimization-bfaf52`
- **总耗时**: 32.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32154231117

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571225) |
| base-b-test-16-npu-a3 / run (0) | 30.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571349) |
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571355) |
| base-b-test-4-npu-a3 / run (0) | 27.9min | 环境问题 | 自定义容器执行失败，NPU后端不支持某些操作导致服务异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571361) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 27.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767572165) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 27.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767572188) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.4min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95769930928) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 7.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95775234720) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅有GitHub Actions环境准备、Node版本警告及上传失败产物（无文件）等常规输出，无法判断测试失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571225

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型权重加载到89%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571349

- **base-a-test-1-npu-a2 / run (0)**: 作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571355

- **base-b-test-4-npu-a3 / run (0)**: 日志显示NPU后端存在不支持的操作回退到CPU，服务启动后health_generate返回503，最终自定义容器执行失败，可能是NPU环境配置或容器问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571361

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在15:55:04时出现错误“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767572165

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在运行约27分钟后，日志显示"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767572188

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于CI流程中的级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95769930928

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95775234720

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571299) |
| base-b-test-2-npu-a3 / run (0) | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571309) |
| base-b-test-4-npu-a3 / run (1) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571436) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571501) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767571997) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32154231117/job/95767572221) |


## [Run #32153734373](https://github.com/sgl-project/sglang/actions/runs/32153734373)
- **分支**: `RM/glm52-ptpc-fp8-proj`
- **总耗时**: 131.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32153734373

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902042) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902284) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95767736014) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95773769777) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95778134626) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95802035503) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902042

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动前的健康检查中检测到multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902284

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95767736014

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95773769777

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95778134626

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤，最终因快速失败策略终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95802035503

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902199) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902280) |
| base-b-test-2-npu-a3 / run (0) | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902297) |
| base-b-test-16-npu-a3 / run (0) | 56.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902389) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902539) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902671) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765902894) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765903023) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765903031) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 117.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32153734373/job/95765903131) |


---
*Auto-generated by npu_pr_monitor.py*