# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-29 10:17 UTC
**本次检查已合入 PR 数**: 31
**涉及 NPU**: 12 | **无关**: 2 | **不确定**: 17

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#40470](https://github.com/sgl-project/sglang/pull/40470) [Diffusion] Add bounded exact conditioning cache across native models
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 51

### [#34528](https://github.com/sgl-project/sglang/pull/34528) [SM120] Add optional FlashInfer PCIe-IPC all-reduce for switch-free hosts
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#39627](https://github.com/sgl-project/sglang/pull/39627) [Radix Cache] Sync Rust TreeCore and make it the default
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 42

### [#41554](https://github.com/sgl-project/sglang/pull/41554) [Refactor] Retire the layer facade and simplify boundary internals
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 100

### [#41552](https://github.com/sgl-project/sglang/pull/41552) [Refactor] Construct independent decoder stage boundaries
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 79

### [#41551](https://github.com/sgl-project/sglang/pull/41551) [Refactor] Select reduction fusion at the consumer
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 44

### [#41550](https://github.com/sgl-project/sglang/pull/41550) [Refactor] Capture auxiliary states at residual reads
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 29

### [#41549](https://github.com/sgl-project/sglang/pull/41549) [Refactor] Carry residual state across stage boundaries
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 73

### [#41548](https://github.com/sgl-project/sglang/pull/41548) [Refactor] Centralize decoder output access
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 52

### [#41527](https://github.com/sgl-project/sglang/pull/41527) [npu]support NPU 910C L2 memcache offload
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#41021](https://github.com/sgl-project/sglang/pull/41021) dsv4.1-amd: fused mHC boundary and all-reduce + mHC post kernels
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#41020](https://github.com/sgl-project/sglang/pull/41020) dsv4.1-amd: gfx950 sparse decode attention and sorted top-k
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 17

## ❓ 不确定是否涉及 NPU 的 PR

### [#375](https://github.com/sgl-project/sglang/pull/375) Reduce overhead when `fork(1)`
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41059](https://github.com/sgl-project/sglang/pull/41059) [Fix] Add name mapping in load_weights of nvidia/LocateAnything-3B
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33804](https://github.com/sgl-project/sglang/pull/33804) [Intel][XPU]Enable chunked prefill scnearios for XPU with UT
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40828](https://github.com/sgl-project/sglang/pull/40828) [XPU] Support compressed-tensors W4A16 by reusing the torch int4pack path
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41618](https://github.com/sgl-project/sglang/pull/41618) [PD] Give FakeKVReceiver ensure_abort_notified
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39385](https://github.com/sgl-project/sglang/pull/39385) [Rust] Extract a transport-neutral frontend core
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41556](https://github.com/sgl-project/sglang/pull/41556) [Refactor] Group layer boundary unit tests
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41555](https://github.com/sgl-project/sglang/pull/41555) [Refactor] Rename the module to layer_boundary
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41553](https://github.com/sgl-project/sglang/pull/41553) [Refactor] Migrate specialized decoder and overlap boundaries
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41557](https://github.com/sgl-project/sglang/pull/41557) [Refactor] Document layer boundary contracts and integration
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41547](https://github.com/sgl-project/sglang/pull/41547) [Refactor] Group communicator fusion and CP adapters
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41600](https://github.com/sgl-project/sglang/pull/41600) [Test] Fail fast when PD test RDMA devices are not openable by ibverbs
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41066](https://github.com/sgl-project/sglang/pull/41066) [Diffusion] Support FLUX 3 Action robot policies
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41469](https://github.com/sgl-project/sglang/pull/41469) [mem cache] refactor: remove the index-K continuous getters orphaned by the CP v1 removal
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41590](https://github.com/sgl-project/sglang/pull/41590) [Model] Add IQuest Q1 support and MTP draft
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41118](https://github.com/sgl-project/sglang/pull/41118) [Docs] Add GigaChat 3.5 and GigaChat 3.5 Reasoning cookbook pages
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41588](https://github.com/sgl-project/sglang/pull/41588) [Model Loader] Stop checkpoint prefetch after iterator completion
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#41464](https://github.com/sgl-project/sglang/pull/41464) [AMD] Fix GLM-5.3 quark MoE MI35x test runner config
- [#41608](https://github.com/sgl-project/sglang/pull/41608) Remove GLM-4.1V-9B-Thinking from encoder DP MMMU test

---
*Auto-generated by npu_pr_monitor.py*