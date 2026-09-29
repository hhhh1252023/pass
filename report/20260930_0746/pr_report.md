# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-29 23:46 UTC
**本次检查已合入 PR 数**: 28
**涉及 NPU**: 6 | **无关**: 4 | **不确定**: 18

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#41756](https://github.com/sgl-project/sglang/pull/41756) [Refactor] Share PD routing fields across request models
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#40977](https://github.com/sgl-project/sglang/pull/40977) [sglang-miles] Cherry-pick #40448, #40648, #40978, #40979, #40980 for MiMo-V2.6-Flash colocated RL
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 32

### [#40693](https://github.com/sgl-project/sglang/pull/40693) [Router] Hold a booting rank's batches and graft a snapshot on the pump (7/13)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#40692](https://github.com/sgl-project/sglang/pull/40692) [Router] Vet a peer's snapshot before it may touch the tree (6/13)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#40690](https://github.com/sgl-project/sglang/pull/40690) [Router] Watch EndpointSlices for sibling router replicas (4/13)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#39313](https://github.com/sgl-project/sglang/pull/39313) fuse shared experts with routed experts in MegaMoE's DeepGEMM
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

## ❓ 不确定是否涉及 NPU 的 PR

### [#41486](https://github.com/sgl-project/sglang/pull/41486) [Perf] Tune SM90 GDN recurrent verify launch for small batches
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41681](https://github.com/sgl-project/sglang/pull/41681) [Test] Demote PD test RDMA openability check to a warning
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40159](https://github.com/sgl-project/sglang/pull/40159) [Spec] Reuse K3 auxiliary outputs across decode CUDA graph sizes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41709](https://github.com/sgl-project/sglang/pull/41709) [Rust] Define semantic frontend contract
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39711](https://github.com/sgl-project/sglang/pull/39711) [PD] Preserve bootstrap metadata in native Messages requests
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39706](https://github.com/sgl-project/sglang/pull/39706) [Metrics] Fix PD latency histogram accounting
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40275](https://github.com/sgl-project/sglang/pull/40275) [Metrics] Add request-level TPOT histogram (sglang:request_time_per_output_token_seconds)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41703](https://github.com/sgl-project/sglang/pull/41703) Fix pip install: exclude multimodal_gen/.claude symlink from package data
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38600](https://github.com/sgl-project/sglang/pull/38600) [metrics] Fix non-streaming TTFT by flushing the first output through detokenization
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#35330](https://github.com/sgl-project/sglang/pull/35330) [Kimi] Enable GB300 TP4 and GB200/GB300 TP16 SP collectives
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41444](https://github.com/sgl-project/sglang/pull/41444) [MUSA] Fix fused MoE GEMV registration and torchada pin
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41650](https://github.com/sgl-project/sglang/pull/41650) [XPU] publish nightly docker image with sgl-kernel-xpu built from main
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40691](https://github.com/sgl-project/sglang/pull/40691) [Router] Track a booting rank's bootstrap and fetch a peer's snapshot (5/13)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41268](https://github.com/sgl-project/sglang/pull/41268) [sgl-router] Add --tokenizer-backend fast and --tokenizer-l1-cache-mb
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41602](https://github.com/sgl-project/sglang/pull/41602) [AMD] Add GLM-5.3 MI30x and MI35x nightly accuracy tests
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41065](https://github.com/sgl-project/sglang/pull/41065) Fix MUSA detection under torch.compile fullgraph
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#28982](https://github.com/sgl-project/sglang/pull/28982) fix(mtp): avoid mtp perf regression in deepseek when enable eplb
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40546](https://github.com/sgl-project/sglang/pull/40546) [AMD] fix kda decode flydsl import
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#41746](https://github.com/sgl-project/sglang/pull/41746) [Docs] Add IQuestLab card logo to cookbook landing page
- [#41728](https://github.com/sgl-project/sglang/pull/41728) chore: grant jain-ria CI permissions
- [#41720](https://github.com/sgl-project/sglang/pull/41720) [CI] Extend DeepGEMM GB300 validation timeout to three hours
- [#40515](https://github.com/sgl-project/sglang/pull/40515) chore: expose agent skills via `.agents` directories

---
*Auto-generated by npu_pr_monitor.py*