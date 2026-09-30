# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-30 23:00 UTC
**本次检查已合入 PR 数**: 42
**涉及 NPU**: 11 | **无关**: 3 | **不确定**: 28

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#40077](https://github.com/sgl-project/sglang/pull/40077) [Fix] Preserve inline instructions for Responses and Kimi K3 Messages
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#41726](https://github.com/sgl-project/sglang/pull/41726) [Rust] Extract shared runtime.v1 protobuf bindings
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 19

### [#41817](https://github.com/sgl-project/sglang/pull/41817) [Refactor] Enter draft TP scopes by attention ownership and drop ModelRunner.tp_group
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 29

### [#41810](https://github.com/sgl-project/sglang/pull/41810) [Fix] Make the Solar model constructible and runnable
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#39012](https://github.com/sgl-project/sglang/pull/39012) [Deps] Bump transformers to 5.17.0
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 21

### [#41308](https://github.com/sgl-project/sglang/pull/41308) dsv4.1-amd: serve DeepSeek-V4.1 on gfx950
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 39

### [#41689](https://github.com/sgl-project/sglang/pull/41689) [diffusion] nightly: measure every framework with one client end-to-end methodology
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#40697](https://github.com/sgl-project/sglang/pull/40697) [Router] Prove a bootstrapped replica answers like the one it copied (11/13)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40696](https://github.com/sgl-project/sglang/pull/40696) [Router] Ask the fleet when a graft's splice goes unwitnessed (10/13)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#41445](https://github.com/sgl-project/sglang/pull/41445) [KDA] Enable the ptx_kda prefill backend on SM100 (B200 / GB200)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#41856](https://github.com/sgl-project/sglang/pull/41856) ci: temporarily disable A5 (950) nightly job during power maintenance
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

## ❓ 不确定是否涉及 NPU 的 PR

### [#41667](https://github.com/sgl-project/sglang/pull/41667) [Fix] Keep MiMo-V2 processor available without TorchCodec
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41896](https://github.com/sgl-project/sglang/pull/41896) [rust-renderer] `sglang-processor` lib
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41154](https://github.com/sgl-project/sglang/pull/41154) fix(spec): enable Qwen3.5 EAGLE3 capture and streaming overlap
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41818](https://github.com/sgl-project/sglang/pull/41818) [Feature] Add --attn-dp-size and deprecate --enable-dp-attention
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#34201](https://github.com/sgl-project/sglang/pull/34201) [RL, Spec] Introduce top-p mask capture for spec and add DFlash/DSpark impl
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41815](https://github.com/sgl-project/sglang/pull/41815) [Refactor] Stop passing models' layers the placement they already read
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41816](https://github.com/sgl-project/sglang/pull/41816) [Refactor] Add make_pp_layers so models stop handing their PP position down
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41814](https://github.com/sgl-project/sglang/pull/41814) [Refactor] Let FusedMoE's weight-loading helpers read the layer's MoE-TP rank
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41813](https://github.com/sgl-project/sglang/pull/41813) [Refactor] Keep only the TP and PP groups on the model runner
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41812](https://github.com/sgl-project/sglang/pull/41812) [Refactor] Drop model placement attributes and parameters nothing reads
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41811](https://github.com/sgl-project/sglang/pull/41811) [Refactor] Drop placement values nothing reads
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41808](https://github.com/sgl-project/sglang/pull/41808) [Fix] Pass the draft's attention ownership to DFLASH's eager LiLiCorr scope
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41809](https://github.com/sgl-project/sglang/pull/41809) [Fix] Size the prefill delayer's gather buffer to its TP group
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41805](https://github.com/sgl-project/sglang/pull/41805) [Fix] Read TensorCast's WORLD placement under its current names
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41807](https://github.com/sgl-project/sglang/pull/41807) [Fix] Shard MoE WNA16 and Quark INT4-FP8 weights by the MoE placement
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41806](https://github.com/sgl-project/sglang/pull/41806) [Fix] State the configured MoE-DP width in the weight-cache fingerprint
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40001](https://github.com/sgl-project/sglang/pull/40001) [Spec][PP] Fix hybrid recurrent-state commit and micro-batch pairing under PP x speculative decoding
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40709](https://github.com/sgl-project/sglang/pull/40709) [Deps] Bump FlashInfer to 0.7.0.post1
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#37984](https://github.com/sgl-project/sglang/pull/37984) [BugFix] Pass token-major Q/K tensors from Gemma-3 to RadixAttention
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41758](https://github.com/sgl-project/sglang/pull/41758) [HiCache] Attribute buffer-mode storage hits against the joint device match
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41451](https://github.com/sgl-project/sglang/pull/41451) [PD][Mamba] fix: free the COW mamba slot of decode requests dropped before preallocation
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41380](https://github.com/sgl-project/sglang/pull/41380) [Fix] Stop PD-decode queue_time from counting decode time before a retraction
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41838](https://github.com/sgl-project/sglang/pull/41838) [Fix] Stop the namespace census at the leaf a read names
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40695](https://github.com/sgl-project/sglang/pull/40695) [Router] Share one fleet-wide fetch across a discovery burst (9/13)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41572](https://github.com/sgl-project/sglang/pull/41572) [KDA] Fix ptx_kda prefill NaN without a gate lower bound and workspace growth
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40911](https://github.com/sgl-project/sglang/pull/40911) [AMD] Gate flashinfer and TRT-LLM DSA paths on CUDA
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40227](https://github.com/sgl-project/sglang/pull/40227) [Linear Attention] Expose GDN/KDA prefill hooks and auxiliary cache accounting
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41513](https://github.com/sgl-project/sglang/pull/41513) [AMD] Resolve QSA packed-varlen decode to aiter on HIP
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#41799](https://github.com/sgl-project/sglang/pull/41799) [Test] Lower the SM120 NVFP4 KV GSM8K threshold to 0.60
- [#41851](https://github.com/sgl-project/sglang/pull/41851) docs: remove unreliable DeepWiki badge
- [#41849](https://github.com/sgl-project/sglang/pull/41849) [Docs] Add MI355X FP8 agentic recipe to the Qwen3.5 cookbook

---
*Auto-generated by npu_pr_monitor.py*