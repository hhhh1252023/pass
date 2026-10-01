# NPU PR 监控报告 (已合入)
**生成时间**: 2026-10-01 09:01 UTC
**本次检查已合入 PR 数**: 41
**涉及 NPU**: 9 | **无关**: 4 | **不确定**: 28

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#41337](https://github.com/sgl-project/sglang/pull/41337) [DSV4/DSA] Name the FlashMLA KV format and drop the V4.1 support probe
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#41265](https://github.com/sgl-project/sglang/pull/41265) [DCP + L3 2/N] Support MLA DCP with Mooncake HiCache L3
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 10

### [#42010](https://github.com/sgl-project/sglang/pull/42010) [Router] Drop the snapshot client's redirect-following fallback
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#41459](https://github.com/sgl-project/sglang/pull/41459) [KDA+Kimi K3] Speed up LTX-2.3 QK norm and split RoPE on H200
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#41999](https://github.com/sgl-project/sglang/pull/41999) [Router] Harden peer-bootstrap edges: refuse redirects and metric docs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#40913](https://github.com/sgl-project/sglang/pull/40913) Declare HiCache host pools for DSA indexer
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 11

### [#40699](https://github.com/sgl-project/sglang/pull/40699) [Router] Gate readiness on peer bootstrap, and make it observable (13/13)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#41789](https://github.com/sgl-project/sglang/pull/41789) [Fix] Emit Responses reasoning content-part events
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#41766](https://github.com/sgl-project/sglang/pull/41766) [Rust] Implement the Rust gRPC adapter for native generation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

## ❓ 不确定是否涉及 NPU 的 PR

### [#41947](https://github.com/sgl-project/sglang/pull/41947) [AMD][V4.1][*/N] Route low-ratio indexer and candidate-block top-k through top-k v2
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41971](https://github.com/sgl-project/sglang/pull/41971) [AMD][V4.1][*/N] Pick the gfx950 wo_a batched gemm tile by row count
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39721](https://github.com/sgl-project/sglang/pull/39721) [Qwen3.8 CP 1/4] Context parallelism for QSA attention and the sparse indexer
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41736](https://github.com/sgl-project/sglang/pull/41736) [DFlash] Keep grouped-conv taps inside each request's block so NaN/Inf cannot leak across requests
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41987](https://github.com/sgl-project/sglang/pull/41987) [DSA] Stop forcing the per-step CPU seq_lens sync for the k-pool indexer
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39788](https://github.com/sgl-project/sglang/pull/39788) [PPU][1/N] CI: Add backend registration and runner preflight
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39166](https://github.com/sgl-project/sglang/pull/39166) [AMD][DSV4] feat: enable PD-disagg with fp8 unified_kv on gfx950
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41610](https://github.com/sgl-project/sglang/pull/41610) [sgl-router] Exclude portless prefills and retry bootstrap discovery
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41924](https://github.com/sgl-project/sglang/pull/41924) [PD] Fix block scale registration for mixed full/SWA KV dtypes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41163](https://github.com/sgl-project/sglang/pull/41163) [DSV4] Reserve the FULL logical page in the c4 indexer pool like its KV pool
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40750](https://github.com/sgl-project/sglang/pull/40750) [AMD] AITER MLA DCP decode ASM path (opt-in)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38991](https://github.com/sgl-project/sglang/pull/38991) [Kimi-K3][DCP][DSpark] fix eager mode nonetype crash due to no extend_prefix_lens
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40471](https://github.com/sgl-project/sglang/pull/40471) fix(dsv4): ship the fp8 unified_kv rope pool through the direct external linker
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41346](https://github.com/sgl-project/sglang/pull/41346) [HiSparse] fix: return HiSparse slots on PD decode waiting abort and DeepSeek V4 free
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41347](https://github.com/sgl-project/sglang/pull/41347) [HiSparse] fix: reject speculative decoding and mixed chunk at startup
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41393](https://github.com/sgl-project/sglang/pull/41393) [HiCache][PD] fix: keep decode load-back polling out of rank-local collectives
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41986](https://github.com/sgl-project/sglang/pull/41986) [Docs] Add NVIDIA NVFP4 checkpoint to GLM-5.3-Flash cookbook
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41837](https://github.com/sgl-project/sglang/pull/41837) Drop the GLM-5.3-Flash breakable prefill chunk-size pin
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41322](https://github.com/sgl-project/sglang/pull/41322) [Fix] Require admin auth for /load_lora_adapter_from_tensors and /set_trace_level
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41972](https://github.com/sgl-project/sglang/pull/41972) [sglang-miles] Sync #40648: drop the DP-attention NCCL graph registration trigger
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41497](https://github.com/sgl-project/sglang/pull/41497) [AMD] Fix MiniMax-M3 EAGLE3 verification on ROCm
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41966](https://github.com/sgl-project/sglang/pull/41966) [diffusion] docs: recommend Docker for GPU deployments
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40698](https://github.com/sgl-project/sglang/pull/40698) [Router] k8s e2e for cache-aware peer bootstrap (12/13)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41936](https://github.com/sgl-project/sglang/pull/41936) [ci] fix renderer image publish
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40811](https://github.com/sgl-project/sglang/pull/40811) [AMD][Quark] Serve the Kimi-K3 MXFP4 checkpoint on ROCm
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41886](https://github.com/sgl-project/sglang/pull/41886) [Perf] Use FA4 by default for MiMo on SM100
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41950](https://github.com/sgl-project/sglang/pull/41950) [Scheduler] Move internal-state readback and updates into a collaborator
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41952](https://github.com/sgl-project/sglang/pull/41952) [Test] Prune redundant scheduler unit tests
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#42018](https://github.com/sgl-project/sglang/pull/42018) [Fix] Let the aiter DCP ASM decode test run without aiter
- [#42023](https://github.com/sgl-project/sglang/pull/42023) [AMD] Update aiter version for v41
- [#41915](https://github.com/sgl-project/sglang/pull/41915) [XPU] Fix test_qsa_indexer_xpu after defer_expansion change
- [#41935](https://github.com/sgl-project/sglang/pull/41935) [ci] grant ci permissions to sagearc

---
*Auto-generated by npu_pr_monitor.py*