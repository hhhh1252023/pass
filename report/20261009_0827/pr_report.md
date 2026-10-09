# NPU PR 监控报告 (已合入)
**生成时间**: 2026-10-09 00:27 UTC
**本次检查已合入 PR 数**: 36
**涉及 NPU**: 17 | **无关**: 3 | **不确定**: 16

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#43180](https://github.com/sgl-project/sglang/pull/43180) [metrics] Count per-rank DP attention pairs and their imbalance
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#43188](https://github.com/sgl-project/sglang/pull/43188) [Bench][MoE] Deal the simulated round-robin experts evenly across EP ranks
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#40177](https://github.com/sgl-project/sglang/pull/40177) [DSV4.1][PD][5/N] Decode node support DP attention
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#43187](https://github.com/sgl-project/sglang/pull/43187) [Metrics] Use widely supported bucket boundaries for the DP attention imbalance ratio
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#43004](https://github.com/sgl-project/sglang/pull/43004) [rust-server] warm up each rust listener in-process
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#43012](https://github.com/sgl-project/sglang/pull/43012) [Speculative] Support block verification for the DFlash family
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#34076](https://github.com/sgl-project/sglang/pull/34076) [Fix] overlap scheduler: record_stream mix_running_indices for the forward-stream relay gather
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#43114](https://github.com/sgl-project/sglang/pull/43114) [http_server] Let custom /generate routes skip the second contract check
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#41072](https://github.com/sgl-project/sglang/pull/41072) [Bugfix] Skip post-experts EP all-reduce on the FlashInfer cutlass FP4 all-gather MoE path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#43013](https://github.com/sgl-project/sglang/pull/43013) [Perf] Fuse short-convolution checkpoint tracking metadata
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#34537](https://github.com/sgl-project/sglang/pull/34537) [AMD][DCP 3/N] add aiter asm for kimi k3 target verify
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#43035](https://github.com/sgl-project/sglang/pull/43035) [NPU] Quote the arm64 Triton-ascend wheel URL in npu.Dockerfile
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#38841](https://github.com/sgl-project/sglang/pull/38841) [MiniMax-M3] Decode: paged K/V tile loads and a tiny GEMM for the router projection
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#38615](https://github.com/sgl-project/sglang/pull/38615) [MiniMax-M3] Instantiate the decode block top-k in register buckets
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#42664](https://github.com/sgl-project/sglang/pull/42664) [rust-processor] DeepSeek-V4 parity with SGLang, plus a reusable parity harness and skill
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 20

### [#43002](https://github.com/sgl-project/sglang/pull/43002) [diffusion] Deduplicate FLUX positions and RoPE application
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#41488](https://github.com/sgl-project/sglang/pull/41488) [AMD] Add opt-in MiniMax-M3 TP4 indexer context partitioning
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

## ❓ 不确定是否涉及 NPU 的 PR

### [#43218](https://github.com/sgl-project/sglang/pull/43218) [metrics] Call DPBalanceStats.create by keyword in the ratio ladder test
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39199](https://github.com/sgl-project/sglang/pull/39199) Add per-token NVFP4 MoE support for ReLU2 activation
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39723](https://github.com/sgl-project/sglang/pull/39723) [Qwen3.8 CP 3/4] Collocated prefill CP integration
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43195](https://github.com/sgl-project/sglang/pull/43195) [CI] Clean some unnecessary DSV4 tests
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42782](https://github.com/sgl-project/sglang/pull/42782) [Model] Add support for EmbeddingGemma 2
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43058](https://github.com/sgl-project/sglang/pull/43058) [CI] Re-enable GB300
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42904](https://github.com/sgl-project/sglang/pull/42904) [PD] Allow state-only KV checksum inputs
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43192](https://github.com/sgl-project/sglang/pull/43192) [Mamba] Describe the track snapshot dtype by its kernel instead of fp32
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43179](https://github.com/sgl-project/sglang/pull/43179) [DFlash] Build the attention QKV projection through an overridable hook
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42690](https://github.com/sgl-project/sglang/pull/42690) [Debug] Add opt-in NaN detection before sampling
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42643](https://github.com/sgl-project/sglang/pull/42643) Stream a chunk once stream_interval tokens are unsent
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43101](https://github.com/sgl-project/sglang/pull/43101) Fix K-only host buffer cleanup
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43100](https://github.com/sgl-project/sglang/pull/43100) Fix DSA index host buffer cleanup
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42903](https://github.com/sgl-project/sglang/pull/42903) [diffusion] Deduplicate request extraction and ComfyUI test helpers
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42665](https://github.com/sgl-project/sglang/pull/42665) [sgl-router] Render DeepSeek-V4 through sglang-processor
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42927](https://github.com/sgl-project/sglang/pull/42927) [Perf] Cache compiled ModelOpt exclusion patterns
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#43226](https://github.com/sgl-project/sglang/pull/43226) [CI] Fix stale fakes in the NVFP4 MoE dispatch and CP strategy unit tests
- [#43022](https://github.com/sgl-project/sglang/pull/43022) [CI] Add dedicated sglang-processor workflow
- [#43025](https://github.com/sgl-project/sglang/pull/43025) [CI] Don't run the full GPU suite for renderer-only Cargo.lock changes

---
*Auto-generated by npu_pr_monitor.py*