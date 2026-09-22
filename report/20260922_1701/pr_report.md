# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-22 09:00 UTC
**本次检查已合入 PR 数**: 48
**涉及 NPU**: 17 | **无关**: 5 | **不确定**: 26

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#39902](https://github.com/sgl-project/sglang/pull/39902) [AMD] Pack Qwen3.5 GDN input projections on ROCm
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#39312](https://github.com/sgl-project/sglang/pull/39312) [observability] Fix negative queue_time for retracted requests
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#39066](https://github.com/sgl-project/sglang/pull/39066) [AMD][Kimi-K3] Fix deferred KDA gate projection and update DCP cookbook
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#40685](https://github.com/sgl-project/sglang/pull/40685) [KDA] Fix missing beta sigmoid in PTX prefill
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#40658](https://github.com/sgl-project/sglang/pull/40658) [DSpark] Fix draft CUDA graph stream explosion
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#40497](https://github.com/sgl-project/sglang/pull/40497) [Docs] GLM-5.3/5.3-Flash cookbooks: enable reasoning/tool-call parsers by default via auto
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#31446](https://github.com/sgl-project/sglang/pull/31446) [HiSparse] Add MHA hisparse support for MiniMax M3
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 27

### [#36120](https://github.com/sgl-project/sglang/pull/36120) [NPU] Fix xgrammar apply_vocab_mask device dispatch to use torch.ops.npu
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#35504](https://github.com/sgl-project/sglang/pull/35504) fix(moe): support Llama4 NVFP4 router input weights on SM120
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40292](https://github.com/sgl-project/sglang/pull/40292) [sgl-router] refactor - SLO ordering for bucket selection
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#40661](https://github.com/sgl-project/sglang/pull/40661) [Test] Handle tied top-k indices in graph-pool logprob regression
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39524](https://github.com/sgl-project/sglang/pull/39524) [Fix] Don't write conv state from the fused KDA verify kernel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40540](https://github.com/sgl-project/sglang/pull/40540) [XPU][ci]: disable XPU NIXL disaggregation test
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40640](https://github.com/sgl-project/sglang/pull/40640) [Kimi K3] Fix CUDA graph stream explosion
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#32673](https://github.com/sgl-project/sglang/pull/32673) [Spec] Windowed draft-decode attention for built-in EAGLE / MTP drafts
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#33778](https://github.com/sgl-project/sglang/pull/33778) Avoid materializing GDN QKV tensors during target verification
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#40642](https://github.com/sgl-project/sglang/pull/40642) [Fix] Run KV canary hooks for context-parallel prefill
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

## ❓ 不确定是否涉及 NPU 的 PR

### [#35872](https://github.com/sgl-project/sglang/pull/35872) [AMD] Skip full-vocab softmax in EAGLE topk==1 draft on ROCm
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39503](https://github.com/sgl-project/sglang/pull/39503) [AMD] Use exact CU share for gfx950 segment-plan headroom
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40111](https://github.com/sgl-project/sglang/pull/40111) avoid host sync in DSpark prefill slot expansion
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39525](https://github.com/sgl-project/sglang/pull/39525) [AMD] Fix deferred Kimi-K3 forget gate in fused in-projection
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39973](https://github.com/sgl-project/sglang/pull/39973) [PD] Validate Mooncake EFA allocator compatibility
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38545](https://github.com/sgl-project/sglang/pull/38545) [AMD] [GLM-5.3-Flash Day 0] Route mHC through AITER on gfx950
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40651](https://github.com/sgl-project/sglang/pull/40651) [LFM2-VL] Add DSpark speculative decoding
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40628](https://github.com/sgl-project/sglang/pull/40628) [ModelOpt][PP] Keep BF16 shared experts out of the NVFP4 fusion so TP1 pipeline stages can load
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39461](https://github.com/sgl-project/sglang/pull/39461) [Router] Abort the engine when a client disconnects mid-request
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40659](https://github.com/sgl-project/sglang/pull/40659) [Benchmark] Optionally clear HiCache storage between cases
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39317](https://github.com/sgl-project/sglang/pull/39317) [AMD] [GLM-5.3-Flash Day 0] Honor fused and per-expert names in quark `exclude`
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40500](https://github.com/sgl-project/sglang/pull/40500) [PD] Pack draft KV head slices for DCP transfers
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38546](https://github.com/sgl-project/sglang/pull/38546) [AMD] [GLM-5.3-Flash Day 0] Enable FP8 and Quark MXFP4 MoE on gfx950
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39486](https://github.com/sgl-project/sglang/pull/39486) perf(engine): avoid timed waits for Engine responses
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40667](https://github.com/sgl-project/sglang/pull/40667) [Test] Set DP size in the mocked Metal profiler test
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39340](https://github.com/sgl-project/sglang/pull/39340) [AMD] [GLM-5.3-Flash Day 0] Support non-2048 top-k widths in the DSA page-table transform
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40590](https://github.com/sgl-project/sglang/pull/40590) [Diffusion] Separate a use-scoped layerwise release from release_all
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40507](https://github.com/sgl-project/sglang/pull/40507) [Diffusion] Restore public Qwen-Image 2.1 TP2 E2E coverage
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40598](https://github.com/sgl-project/sglang/pull/40598) [AMD][Fix] AgentX HIP TPOT regression when SGLANG_SIMULATE_ACC_LEN is set
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40310](https://github.com/sgl-project/sglang/pull/40310) Support GLM-5.3-Flash hybrid attention CPU offload and PD index mapping
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33520](https://github.com/sgl-project/sglang/pull/33520) [Intel][XPU][KVCanary] Enable KV Canary on Intel XPU
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38875](https://github.com/sgl-project/sglang/pull/38875) [AMD] Pad QSA MQA decode Q-heads to 16 for ROCm MFMA
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39338](https://github.com/sgl-project/sglang/pull/39338) [AMD] [GLM-5.3-Flash Day 0] Enable zero-RoPE MHA prefill on ROCm
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40646](https://github.com/sgl-project/sglang/pull/40646) [Fix] Keep diffusion encoder TP context bindings consistent
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40371](https://github.com/sgl-project/sglang/pull/40371) [NPU][BugFix] Avoid M-RoPE recompilation for variable sequence lengths
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#37507](https://github.com/sgl-project/sglang/pull/37507) [unified-memory] Hierarchical cache for every unified pool shape
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#40647](https://github.com/sgl-project/sglang/pull/40647) [DOC] Update quickstart guide to use `sglang serve` for launching the server
- [#40684](https://github.com/sgl-project/sglang/pull/40684) [router] Decode tagged-map KV events alongside legacy tagged arrays
- [#39965](https://github.com/sgl-project/sglang/pull/39965) [AMD] Update ROCm AITER pin to acf8fdf9
- [#40604](https://github.com/sgl-project/sglang/pull/40604) [sgl-router] Fix readiness, IPv6 discovery, logging, and model validation
- [#40654](https://github.com/sgl-project/sglang/pull/40654) Fix lint failure from draft-decode window test location

---
*Auto-generated by npu_pr_monitor.py*