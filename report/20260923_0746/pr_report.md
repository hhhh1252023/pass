# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-22 23:46 UTC
**本次检查已合入 PR 数**: 36
**涉及 NPU**: 13 | **无关**: 5 | **不确定**: 18

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#39632](https://github.com/sgl-project/sglang/pull/39632) fix(function_call): buffer complete DeepSeek DSML invokes
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#40357](https://github.com/sgl-project/sglang/pull/40357) [MM] Keep scheduler padding in packed token arrays
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 64

### [#39779](https://github.com/sgl-project/sglang/pull/39779) [AMD] [GLM-5.3-Flash Day 0] Load the MXFP4 MTP draft layer
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#38547](https://github.com/sgl-project/sglang/pull/38547) [AMD] [GLM-5.3-Flash Day 0] Enable zero-RoPE TileLang DSA on gfx950
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#40686](https://github.com/sgl-project/sglang/pull/40686) [Router] Keep e2e workers inside the job's CUDA_VISIBLE_DEVICES allotment
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#40707](https://github.com/sgl-project/sglang/pull/40707) Take the model config out of the parallel group build, and finish retiring the parallel getters
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 17

### [#40260](https://github.com/sgl-project/sglang/pull/40260) [Feature] Support --tokenizer-worker-num > 1 in the offline Engine API
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40352](https://github.com/sgl-project/sglang/pull/40352) [DSv4.1] Score prefill consumer index layers on candidate blocks with DeepGEMM
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#40133](https://github.com/sgl-project/sglang/pull/40133) [NPU][CI] Constrain evalscope dependency versions
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#38468](https://github.com/sgl-project/sglang/pull/38468) [kv-shard 3/4] Enable Control plane
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#40714](https://github.com/sgl-project/sglang/pull/40714) [NPU] [DOC] Remove duplicated features in npu docs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40175](https://github.com/sgl-project/sglang/pull/40175) [diffusion] attention: add fp8_fa_sm120 FP8 backend for SM120 GPUs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#40557](https://github.com/sgl-project/sglang/pull/40557) [AMD] Drop the redundant scale zero-fill before AITER per-tensor FP8 quant
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

## ❓ 不确定是否涉及 NPU 的 PR

### [#40780](https://github.com/sgl-project/sglang/pull/40780) [mem_cache] Clean up SWA/Mamba radix cache leftovers and drop SGLANG_ENABLE_UNIFIED_RADIX_TREE
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40644](https://github.com/sgl-project/sglang/pull/40644) fix(grpc): expose native response timeout as a server argument
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40775](https://github.com/sgl-project/sglang/pull/40775) [mem_cache] Remove the experimental C++ radix tree
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39341](https://github.com/sgl-project/sglang/pull/39341) [AMD] [GLM-5.3-Flash Day 0] Enable the k-pool DSA indexer on gfx950
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39778](https://github.com/sgl-project/sglang/pull/39778) [AMD] [GLM-5.3-Flash Day 0] Enable speculative decoding (MTP) on ROCm
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40637](https://github.com/sgl-project/sglang/pull/40637) [Fix] Handle chunked paged MQA metadata in DSV4.1 eager forwards
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39901](https://github.com/sgl-project/sglang/pull/39901) [AMD] Reuse KV gather indices across ASM context prefill layers
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40680](https://github.com/sgl-project/sglang/pull/40680) [HiCache] Demote internal-node mamba states on write_back eviction
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40747](https://github.com/sgl-project/sglang/pull/40747) [rust-renderer] decouple renderer sampling from protocols
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40639](https://github.com/sgl-project/sglang/pull/40639) [ci] publish renderer image
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40636](https://github.com/sgl-project/sglang/pull/40636) [ci] run cpu ci for renderer-only changes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40501](https://github.com/sgl-project/sglang/pull/40501) [Qwen3.8-Next] Pipeline-parallel serving and PD-prefill MTP for Qwen4-Exp
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40593](https://github.com/sgl-project/sglang/pull/40593) [Diffusion] Correct the resident-layer help text to match its scope
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40309](https://github.com/sgl-project/sglang/pull/40309) [Fix] Missing SWA eviction during decode preallocation
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40592](https://github.com/sgl-project/sglang/pull/40592) [diffusion] feat: allow a component use retain its layerwise resident set
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40711](https://github.com/sgl-project/sglang/pull/40711) [PD] Simplify late-abort quiescent ack branch to else
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40645](https://github.com/sgl-project/sglang/pull/40645) [PD] Preserve abort ACKs until in-flight KV transfers drain
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40641](https://github.com/sgl-project/sglang/pull/40641) [AMD][DI] Keep loopback in UCX_NET_DEVICES
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#40770](https://github.com/sgl-project/sglang/pull/40770) Add GB200/GB300 hardware to Qwen3.5
- [#40655](https://github.com/sgl-project/sglang/pull/40655) docs: sync LMSYS SGLang blog cards
- [#40114](https://github.com/sgl-project/sglang/pull/40114) [Docs] Fix benchmark table column overflow in cookbook deployment panel
- [#40725](https://github.com/sgl-project/sglang/pull/40725) Fix the Inkling per-expert sync test and collect it in the weekly CPU run
- [#40123](https://github.com/sgl-project/sglang/pull/40123) [AMD] Register unified KV page-zeroing test in PR CI

---
*Auto-generated by npu_pr_monitor.py*