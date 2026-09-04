# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-04 10:10 UTC
**本次检查已合入 PR 数**: 42
**涉及 NPU**: 8 | **无关**: 34 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#35176](https://github.com/sgl-project/sglang/pull/35176) [AMD] [Kimi-K3] Fuse the KDA input projection into a single GEMM on ROCm
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#35751](https://github.com/sgl-project/sglang/pull/35751) [XPU] Support GPT-OSS MXFP4 checkpoints on Intel XPU
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#36223](https://github.com/sgl-project/sglang/pull/36223) [CP V1 Deprecation 2/5] Make strategy prefill CP canonical
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#33824](https://github.com/sgl-project/sglang/pull/33824) [Simulator] Add high-fidelity CPU-based inference simulator
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 81

### [#37820](https://github.com/sgl-project/sglang/pull/37820) [CI] Build the Rust extensions for aarch64 too
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#36320](https://github.com/sgl-project/sglang/pull/36320) [Diffusion] Fix MiniMax H3 WebUI inference settings
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#36843](https://github.com/sgl-project/sglang/pull/36843) [NPU] fix extend_seq_lens_cpu shape in eager mode
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#37874](https://github.com/sgl-project/sglang/pull/37874) [PD] Bound transfer engine init with `SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT`
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

## ✅ 与 NPU 无关的已合入 PR
- [#37876](https://github.com/sgl-project/sglang/pull/37876) [mem_cache] Route hybrid SWA full-side kv-row frees through `free_segment`
- [#35255](https://github.com/sgl-project/sglang/pull/35255) Fix: abort handling for dispatched requests after client disconnect
- [#37658](https://github.com/sgl-project/sglang/pull/37658) [AMD][DSv4] Fuse inverse-RoPE into the fp8 wo_a quant (stacked on #37423)
- [#37836](https://github.com/sgl-project/sglang/pull/37836) Fix mamba radix cache ssm state indexing
- [#37890](https://github.com/sgl-project/sglang/pull/37890) [Diffusion] Improve BCG warmup frame-count diagnostics for video models
- [#37580](https://github.com/sgl-project/sglang/pull/37580) [AMD] Skip unused TOPK v2 plan kernel on ROCm
- [#37423](https://github.com/sgl-project/sglang/pull/37423) [AMD][DSv4] Switch output projection gemm (oproj_a) to fp8
- [#37764](https://github.com/sgl-project/sglang/pull/37764) [AMD][DSv4] Fuse the DSv4 FP4 indexer prefill-schedule preamble into one kernel
- [#37910](https://github.com/sgl-project/sglang/pull/37910) [Diffusion] Fuse LingBot per-token gated residual and RMSNorm modulate
- [#32902](https://github.com/sgl-project/sglang/pull/32902) [Bugfix] Fix Llama 4 FA3 local attention with paged KV cache
- [#37119](https://github.com/sgl-project/sglang/pull/37119) [AMD] CI: fix Lean decode crash on the EAGLE path
- [#35092](https://github.com/sgl-project/sglang/pull/35092) [AMD] Fix DSV4 unified attention sink TP slice
- [#37937](https://github.com/sgl-project/sglang/pull/37937) [Fix] Register triton.runtime.cache.triton_key in the MPS stub so torch.compile keeps working
- [#37891](https://github.com/sgl-project/sglang/pull/37891) [Diffusion][Docs] Add single-GPU large-VRAM performance notes (B300)
- [#37804](https://github.com/sgl-project/sglang/pull/37804) [diffusion] Reduce hot-path server log noise
- [#37829](https://github.com/sgl-project/sglang/pull/37829) [AMD] Update v4 amd cookbook 0903
- [#37930](https://github.com/sgl-project/sglang/pull/37930) [CI] Fix handle_platform_cp_compatibility reading legacy CP flags off the record
- [#37532](https://github.com/sgl-project/sglang/pull/37532) [XPU][CI] Move XPU tests to nightly and add per-subclass server launch timeout
- [#37206](https://github.com/sgl-project/sglang/pull/37206) [Comm] Drop the in-tree MNNVL CuTe DSL port in favor of FlashInfer 0.6.18
- [#37922](https://github.com/sgl-project/sglang/pull/37922) Add code owner for sglang-simulator
- [#37805](https://github.com/sgl-project/sglang/pull/37805) [diffusion] Remove unreachable Cosmos3 transfer encoding
- [#37887](https://github.com/sgl-project/sglang/pull/37887) Allow the GDN out_proj LoRA target in the CLI
- [#37395](https://github.com/sgl-project/sglang/pull/37395) [CPU][CI]: rename Xeon CPU CI suites to stage-*-intel
- [#37895](https://github.com/sgl-project/sglang/pull/37895) [CI] Fix lint
- [#33994](https://github.com/sgl-project/sglang/pull/33994) [diffusion] MiniMax-H3: Only use fused qk_norm on NV
- [#36380](https://github.com/sgl-project/sglang/pull/36380) Cosmos3 fp8 mixed precision
- [#37687](https://github.com/sgl-project/sglang/pull/37687) [ci] Remove MINIMAX_H3_HF_TOKEN
- [#37872](https://github.com/sgl-project/sglang/pull/37872) Fix UNO test adapter subdirectory resolution
- [#37835](https://github.com/sgl-project/sglang/pull/37835) [Diffusion] MiniMax-H3 VAE decoder: unfused w2 bias on SM12.x (cuBLAS 16x16 kernel mis-dispatch)
- [#37816](https://github.com/sgl-project/sglang/pull/37816) [diffusion] Compose third-party component bundles safely
- [#37285](https://github.com/sgl-project/sglang/pull/37285) state_capturer: pin the exact host-cache size via mmap + cudaHostRegister
- [#37884](https://github.com/sgl-project/sglang/pull/37884) [CI] Improve Lark CI cards: structured layout, PDT timestamps, slow-only queue digest
- [#37883](https://github.com/sgl-project/sglang/pull/37883) [HiCache] Count hit allocations and in-flight backups in the buffer pipeline idle check
- [#37503](https://github.com/sgl-project/sglang/pull/37503) [HiCache] L3 storage prefetch lifecycle metrics and cross-tier attribution fixes

---
*Auto-generated by npu_pr_monitor.py*