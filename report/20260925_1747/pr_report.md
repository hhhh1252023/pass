# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-25 09:46 UTC
**本次检查已合入 PR 数**: 31
**涉及 NPU**: 15 | **无关**: 1 | **不确定**: 15

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#41200](https://github.com/sgl-project/sglang/pull/41200) [Refactor] Decide an FFN exit's completion once and declare the group it owes
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#41199](https://github.com/sgl-project/sglang/pull/41199) [Refactor] Take a layer's last-layer fact from its scatter-mode plan
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 28

### [#41191](https://github.com/sgl-project/sglang/pull/41191) [Refactor] Build prepare_mlp and the layout moves from named steps
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#41198](https://github.com/sgl-project/sglang/pull/41198) [Refactor] Move Step-3.5, GLM5-Next, Dots3, MiniMax-M3 and Qwen3.5 onto ffn_exit
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#41197](https://github.com/sgl-project/sglang/pull/41197) [Refactor] Leave the FFN reduction to the next layer under attention DP
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#41196](https://github.com/sgl-project/sglang/pull/41196) [Refactor] Carry a deferred FFN all-reduce as UnreducedOutput and complete it in the next layer without the fused kernel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 27

### [#41194](https://github.com/sgl-project/sglang/pull/41194) [Fix] Plan NextN / MTP draft layers as one-layer models and fix the Bailing V2 NextN draft
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#41193](https://github.com/sgl-project/sglang/pull/41193) [Fix] Complete the all-reduce when the flashinfer fused norm declines a batch
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#41074](https://github.com/sgl-project/sglang/pull/41074) fix: load fused shared-expert LoRA weights and bound expert indices
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#41208](https://github.com/sgl-project/sglang/pull/41208) [Feature] Add a System One compatible /v1/systemone route
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 19

### [#41062](https://github.com/sgl-project/sglang/pull/41062) [Fix] Keep the target's DP sync slot in draft scopes
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 10

### [#41095](https://github.com/sgl-project/sglang/pull/41095) [diffusion] Add opt-in SRT prompt enhancement to image and video APIs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 11

### [#41091](https://github.com/sgl-project/sglang/pull/41091) [DSV4] Account for FlashMLA physical KV page padding in memory budgets
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40907](https://github.com/sgl-project/sglang/pull/40907) [AMD] Restore non-DCP Mamba checkpoint donation to fix agent-mode cache hit at high conc with HiCache
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#38965](https://github.com/sgl-project/sglang/pull/38965) [Score API] Setwise Scoring Support
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 18

## ❓ 不确定是否涉及 NPU 的 PR

### [#41224](https://github.com/sgl-project/sglang/pull/41224) [XPU] Disable test_ngram_corpus on XPU and extend XPU CI path filter
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41195](https://github.com/sgl-project/sglang/pull/41195) [Fix] Stop counting a deferred FFN sum more than once: replicated TP1 shared expert, dense reduce_scatterv
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#35619](https://github.com/sgl-project/sglang/pull/35619) [AMD] Integrate Aiter MegaMoEv2 for DeepSeek-V4
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41092](https://github.com/sgl-project/sglang/pull/41092) [HiCache] fix: Drain pending backups before internal Mamba write-back
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40456](https://github.com/sgl-project/sglang/pull/40456) [HiCache] Give trailing sidecar storage transfers a contiguous prefix_keys chain
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41067](https://github.com/sgl-project/sglang/pull/41067) [diffusion] Add native Ming-Image Design and Design-Layer support
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41201](https://github.com/sgl-project/sglang/pull/41201) [DeepEP v2] Let a model package supply its per-rank prefill dispatch bound
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41159](https://github.com/sgl-project/sglang/pull/41159) [AMD] Fix int32 offset overflow in Triton DSv4 KV store kernels
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#28135](https://github.com/sgl-project/sglang/pull/28135) fix(openai): reject request-supplied chat_template by default
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41061](https://github.com/sgl-project/sglang/pull/41061) [Perf] Lazy-load built-in model definitions and nixl_ep at startup
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40793](https://github.com/sgl-project/sglang/pull/40793) [PD] Honor gracefully_exit in disaggregation event loops and keep non-zero-rank launchers alive on SIGTERM
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40821](https://github.com/sgl-project/sglang/pull/40821) [sglang-miles] Colocated RL for hybrid-state models: flush order and bf16 MoE weight-update layout
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40988](https://github.com/sgl-project/sglang/pull/40988) [mem_cache] Drop `is_insert` from `cache_finished_req`; release rows from `release_kv_cache`
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41103](https://github.com/sgl-project/sglang/pull/41103) [PD] Add a `none` decode retraction backup and subclass seams in the PD queues
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40960](https://github.com/sgl-project/sglang/pull/40960) [HiCache] Batch buffer-only KV backups within each flush
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#41207](https://github.com/sgl-project/sglang/pull/41207) [CI] Move GLM-5.2 layer-split test to extra-b-test-8-gpu-b300

---
*Auto-generated by npu_pr_monitor.py*