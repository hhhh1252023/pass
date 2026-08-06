# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-06 08:24 UTC
**本次检查已合入 PR 数**: 33
**涉及 NPU**: 7 | **无关**: 5 | **不确定**: 21

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#33775](https://github.com/sgl-project/sglang/pull/33775) [diffusion] feat: capture-safe pynccl all-to-all
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#33753](https://github.com/sgl-project/sglang/pull/33753) [AMD] [CI] Track the MI355X disagg nightly in the AMD CI job monitor
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#33490](https://github.com/sgl-project/sglang/pull/33490) config: retire ServerArgs.override in favour of derive()
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 22

### [#33214](https://github.com/sgl-project/sglang/pull/33214) Fix DeepSeek-OCR batching crash on variable local-crop counts
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#33564](https://github.com/sgl-project/sglang/pull/33564) Fix Nightly NV CI
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#33148](https://github.com/sgl-project/sglang/pull/33148) [Quantization] Route per-tensor FP8 checkpoints to FlashInfer on SM90
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#33205](https://github.com/sgl-project/sglang/pull/33205) [Kernel] Unify BaseFusedOp and MultiPlatformOp dispatch
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 23

## ❓ 不确定是否涉及 NPU 的 PR

### [#33830](https://github.com/sgl-project/sglang/pull/33830) Fix lint error
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33694](https://github.com/sgl-project/sglang/pull/33694)  [AMD] Gate DFLASH non-greedy verify on the target-only kernel being registered
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33826](https://github.com/sgl-project/sglang/pull/33826) Revert "Warn on risky serving-time Triton work"
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33734](https://github.com/sgl-project/sglang/pull/33734) [diffusion] ERNIE-Image bit-exact residual-gate fast path (H200 1024^2 e2e 16.17 -> 15.75 s)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33725](https://github.com/sgl-project/sglang/pull/33725) [diffusion] feat: data-parallel serving (--dp-size)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33655](https://github.com/sgl-project/sglang/pull/33655) [diffusion] Prefer cuDNN SDPA over FA4 for dense attention on sm_100 (B200)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33776](https://github.com/sgl-project/sglang/pull/33776) [CI] Bound the CUDA graph capture range in test launches and lift the spec fixture's admission cap
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33780](https://github.com/sgl-project/sglang/pull/33780) Relax GDN ReplaySSM fold test for Triton 3.7
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33491](https://github.com/sgl-project/sglang/pull/33491) config: resolve the draft worker's config per runner, not on a copy
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33492](https://github.com/sgl-project/sglang/pull/33492) config: the draft runner carries its own attention backend
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33489](https://github.com/sgl-project/sglang/pull/33489) config: template-detected parsers go to the engine's control-plane overlay
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33488](https://github.com/sgl-project/sglang/pull/33488) config: retire the alias-form process-global config reads
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33487](https://github.com/sgl-project/sglang/pull/33487) config: pass the Ray placement group as a launch argument
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#27692](https://github.com/sgl-project/sglang/pull/27692) [RL] Skip rotary cache tensors in weight checker
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33748](https://github.com/sgl-project/sglang/pull/33748) [Fix] Vocab out of bounds in DSpark for Inkling-Small
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33469](https://github.com/sgl-project/sglang/pull/33469) kernels: scalar scale A support for fp8_gemm
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33779](https://github.com/sgl-project/sglang/pull/33779) [Cherry-pick to release/v0.5.17] Fix Nightly NV CI (#33564)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33750](https://github.com/sgl-project/sglang/pull/33750) [Fix] Inkling works with gs:// runai_streamer paths
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33774](https://github.com/sgl-project/sglang/pull/33774) [AMD]Stage MI355X nightly by node count
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33618](https://github.com/sgl-project/sglang/pull/33618) Enable MoE deferred finalize by default and drop its expert_weights dtype workaround
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33678](https://github.com/sgl-project/sglang/pull/33678) chore: bump sgl-kernel version to 0.4.6
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#33138](https://github.com/sgl-project/sglang/pull/33138) Implement random tie breadking for cache_aware sglang router policy
- [#33825](https://github.com/sgl-project/sglang/pull/33825) [AMD] Update amd k3 cookbook for fp8 kv cache
- [#33738](https://github.com/sgl-project/sglang/pull/33738) docker: pin nightly image source to workflow commit
- [#33786](https://github.com/sgl-project/sglang/pull/33786) [Cherry-pick to release/v0.5.17] [CI] Temporarily disable prefill cuda graph for qwen3.5 nightly test (#33772)
- [#33772](https://github.com/sgl-project/sglang/pull/33772) [CI] Temporarily disable prefill cuda graph for qwen3.5 nightly test

---
*Auto-generated by npu_pr_monitor.py*