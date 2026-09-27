# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-27 10:30 UTC
**本次检查已合入 PR 数**: 67
**涉及 NPU**: 27 | **无关**: 2 | **不确定**: 38

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#41305](https://github.com/sgl-project/sglang/pull/41305) [KDA+Kimi K3] Speed up SANA-Video residual gate add on H200
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#41221](https://github.com/sgl-project/sglang/pull/41221) [sgl-router] Take SGLang's render defaults: --default-chat-template-kwargs and thinking/effort envs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 31

### [#41325](https://github.com/sgl-project/sglang/pull/41325) Make sliding-window caching and speculative batch padding extensible
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 17

### [#41342](https://github.com/sgl-project/sglang/pull/41342) [sgl-router] Book the input_ids forwarding outcome only for built bodies
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#41261](https://github.com/sgl-project/sglang/pull/41261) [PD] fix: cache resumed decode-radix requests from root instead of an unlocked re-match
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#41256](https://github.com/sgl-project/sglang/pull/41256) [Refactor] Pick the two-batch-overlap split's layout moves once and remove execute
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#41257](https://github.com/sgl-project/sglang/pull/41257) [Refactor] Choose a dense layer's boundaries under attention DP from both sides' declarations
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 11

### [#41255](https://github.com/sgl-project/sglang/pull/41255) [Refactor] Run the LayerNorm SP region's boundary steps in the communicator itself
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#41252](https://github.com/sgl-project/sglang/pull/41252) [Refactor] Choose prepare_attn / prepare_mlp steps and fused kernels at construction
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 10

### [#41188](https://github.com/sgl-project/sglang/pull/41188) [Score API] Setwise scoring: CausalLM support (batched + --enable-mis)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 10

### [#41125](https://github.com/sgl-project/sglang/pull/41125) [DSv4.1] Move the low-ratio index top-k into dsv4/low_ratio_indexer
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 18

### [#41185](https://github.com/sgl-project/sglang/pull/41185) [sgl-router] Track input_ids forwarding outcomes per chat request
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#41243](https://github.com/sgl-project/sglang/pull/41243) [Refactor] Restore logical kernel groups and test organization
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 100

### [#36505](https://github.com/sgl-project/sglang/pull/36505) [ROCm][Perf] aiter: page-level KV view for gfx950 fp8 page-64 asm prefill
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#40854](https://github.com/sgl-project/sglang/pull/40854) [DSA] Chunk the kpool indexer MQA logits by query rows under a free-memory budget
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40490](https://github.com/sgl-project/sglang/pull/40490) [diffusion] Fuse lossless Klein packed QK RMSNorm and RoPE on Hopper
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#41285](https://github.com/sgl-project/sglang/pull/41285) [CI] Harden `/rerun-test` dispatch and partitioning
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#41018](https://github.com/sgl-project/sglang/pull/41018) dsv4.1-amd: gfx950 MXFP8 matmul kernels and fp8-grid producers
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 10

### [#39660](https://github.com/sgl-project/sglang/pull/39660) [PD] Share one head-slice helper across mooncake, mori, and nixl
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#38652](https://github.com/sgl-project/sglang/pull/38652) [LMCache] Support lmcache unified radix cache
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 20

### [#39379](https://github.com/sgl-project/sglang/pull/39379) [LoRA] Size dense row/column-parallel LoRA buffers from the base linear's real shard
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40786](https://github.com/sgl-project/sglang/pull/40786) [Diffusion][sglang-miles] Bring sglang-miles-h3 H3 rollout commits onto sglang-miles + LoRA-wrap fixes
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 17

### [#41246](https://github.com/sgl-project/sglang/pull/41246) [Rust frontend] Decode input_ids without untagged buffering
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#40986](https://github.com/sgl-project/sglang/pull/40986) [Sampling] Stream sampling masks as per-request arrays
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 18

### [#40621](https://github.com/sgl-project/sglang/pull/40621) Fix multimodal feature offload races
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#28131](https://github.com/sgl-project/sglang/pull/28131) fix(multimodal): return 400 for corrupt image inputs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#41132](https://github.com/sgl-project/sglang/pull/41132) Revert "[NPU] Fuse FIA KV-cache K/V writes into one npu_scatter_pa_kv_cache call"
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

## ❓ 不确定是否涉及 NPU 的 PR

### [#41166](https://github.com/sgl-project/sglang/pull/41166) [qwen 3.8 next] Fuse small CUDA graph input buffer copies
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39964](https://github.com/sgl-project/sglang/pull/39964) [kv-shard 3/4] Enable Control Plane B
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41345](https://github.com/sgl-project/sglang/pull/41345) [DSV4.1][HiCache] fix: wait for the layer transfer before reading low-ratio index-K
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#34416](https://github.com/sgl-project/sglang/pull/34416) [Diffusion] Preserve per-sample rollout trajectories across multi-output merge
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41312](https://github.com/sgl-project/sglang/pull/41312) [mem_cache] Never free the protected prefix on request release
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41019](https://github.com/sgl-project/sglang/pull/41019) dsv4.1-amd: KV cache layouts, FP4 indexer, compressor and router kernels
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41377](https://github.com/sgl-project/sglang/pull/41377) [AMD] Honor an explicit triton moe_runner_backend for mxfp8 on ROCm
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41270](https://github.com/sgl-project/sglang/pull/41270) [cherrypick from #40986] [Sampling] Stream sampling masks as per-request arrays
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41328](https://github.com/sgl-project/sglang/pull/41328) [MemCache] Fix LMCache component cursors and per-cache backend selection
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41356](https://github.com/sgl-project/sglang/pull/41356) [AMD] Fix jit broken on rocm env
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41292](https://github.com/sgl-project/sglang/pull/41292) Pin triton_kernels num_warps for MXFP4 MoE below Hopper (6x gpt-oss decode on RTX 4090)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41321](https://github.com/sgl-project/sglang/pull/41321) [CI] Merge the Kimi-Linear PD DCP4 nightly tests and drop exact-token parity
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41271](https://github.com/sgl-project/sglang/pull/41271) [cherrypick from #41235] [PD] Keep the sampling mask of a replayed rebootstrap token
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41349](https://github.com/sgl-project/sglang/pull/41349) [CI] Move DeepSelect into the attention kernel group to fix the namespace test
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41341](https://github.com/sgl-project/sglang/pull/41341) [CI] Add __main__ entry to test_deep_select.py
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41254](https://github.com/sgl-project/sglang/pull/41254) [Refactor] Declare at construction the layers whose FFN completes its own reduction
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41253](https://github.com/sgl-project/sglang/pull/41253) [Refactor] Drive an FFN exit's flags and its completion from one selection
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40556](https://github.com/sgl-project/sglang/pull/40556) [DeepSeek V4.1] Add DeepSelect JIT kernel.
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41223](https://github.com/sgl-project/sglang/pull/41223) [Perf] Mamba2 selective_state_update up to 2x faster on B200 via 8x1 launch config for dstate 128 (+7.5% Nemotron-3-Super serving)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41311](https://github.com/sgl-project/sglang/pull/41311) [DSA] Fix the pooled-indexer breakable prefill bridge under DP attention
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41298](https://github.com/sgl-project/sglang/pull/41298) [diffusion] Copy small files into overlay materialized trees instead of linking them
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41150](https://github.com/sgl-project/sglang/pull/41150) [Bug][Diffusion] Fix Qwen-Image 2.1 default RGBA output
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41276](https://github.com/sgl-project/sglang/pull/41276) [MemCache] Unify component eviction cursors and lock receipts
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41297](https://github.com/sgl-project/sglang/pull/41297) [Test] Remove more unit tests that mirror implementation or never run in CI
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41011](https://github.com/sgl-project/sglang/pull/41011) [diffusion] Add native Anima Base v1.0 support
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41291](https://github.com/sgl-project/sglang/pull/41291) [DSv4.1] Move the ratio-1/2 index top-k ops into kernels/ops/attention/dsv4
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39731](https://github.com/sgl-project/sglang/pull/39731) [DCP] Use logical token capacity for PD admission and load reporting
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41286](https://github.com/sgl-project/sglang/pull/41286) [Test] Remove unit tests that only mirror implementation or never run in CI
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41284](https://github.com/sgl-project/sglang/pull/41284) [misc] Call all_gather_single / reduce_scatter_single to drop torch deprecation warnings
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41280](https://github.com/sgl-project/sglang/pull/41280) [Test] Route all sgl-eval benchmarks through run_sgl_eval and deprecate run_eval
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#36389](https://github.com/sgl-project/sglang/pull/36389) [sglang][lora] Support DP attention in LoRA backends
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41216](https://github.com/sgl-project/sglang/pull/41216) [Test] Add in-process sgl-eval adapter and move validated GSM8K tests to it
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41215](https://github.com/sgl-project/sglang/pull/41215) [Test] Remove dead eval modules and point GSM8K/MMLU docs to sgl-eval
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41248](https://github.com/sgl-project/sglang/pull/41248) [unified-memory] Honor move gates in float relocation and size auto HiCache from host capacity
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39064](https://github.com/sgl-project/sglang/pull/39064) [ROCm][Bugfix] Keep quantization for mixed Quark Qwen3.5 MTP checkpoints
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41247](https://github.com/sgl-project/sglang/pull/41247) Fix KV offload synchronization
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#34417](https://github.com/sgl-project/sglang/pull/34417) [Diffusion] Fix AttributeError in grouped forward_batch by installing the residency manager
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40486](https://github.com/sgl-project/sglang/pull/40486) [diffusion] Enable lossless Cosmos3 Super T2I QK fusion on Hopper TP2
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#11](https://github.com/sgl-project/sglang/pull/11) Update Readme
- [#41287](https://github.com/sgl-project/sglang/pull/41287) [Fix] Fix cpu CI fail introduced by pr #38652

---
*Auto-generated by npu_pr_monitor.py*