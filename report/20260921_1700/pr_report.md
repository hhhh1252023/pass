# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-21 09:00 UTC
**本次检查已合入 PR 数**: 28
**涉及 NPU**: 11 | **无关**: 1 | **不确定**: 16

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#40544](https://github.com/sgl-project/sglang/pull/40544) [NPU][Diffusion] Disable loading latency checks in Ascend fixtures
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#40271](https://github.com/sgl-project/sglang/pull/40271) [sgl-router] refactor - generalized admission policy definitions
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#40549](https://github.com/sgl-project/sglang/pull/40549) [NPU][CI] Fix paths-filter negation that makes every PR run the NPU tier
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39956](https://github.com/sgl-project/sglang/pull/39956) [ci][xpu] Record device time in the multimodal_gen perf lane
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#38786](https://github.com/sgl-project/sglang/pull/38786) [Fix] Preserve YaRN scaling when extending rotary caches
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#32792](https://github.com/sgl-project/sglang/pull/32792) [XPU]Enable HiSparse hierarchical sparse KV cache on Intel XPU
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#38779](https://github.com/sgl-project/sglang/pull/38779) Fix: post-load staging regression breaks offload meta/sharded_gpu modes
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40201](https://github.com/sgl-project/sglang/pull/40201) [Perf] Fork-safe import: no CUDA context at import time, lighter argument parsing
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 11

### [#36187](https://github.com/sgl-project/sglang/pull/36187) [npu]add chunk gdn kernel and unify ssm state layout for ascend gdn backend
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#40481](https://github.com/sgl-project/sglang/pull/40481) [Diffusion] Reduce Qwen-Image 2.1 VAE and graph warmup memory
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#40241](https://github.com/sgl-project/sglang/pull/40241) [sgl-router] refactor - layout BucketResolver, Bucket, EngineGroup and implement PowerOfTwo
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 15

## ❓ 不确定是否涉及 NPU 的 PR

### [#40113](https://github.com/sgl-project/sglang/pull/40113) [AMD][DI][CI] Add a SPUR cluster profile to AMD DI CI 
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40537](https://github.com/sgl-project/sglang/pull/40537) [sgl-router] Fix reorg admission proxy test build after BucketResolver::new
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40554](https://github.com/sgl-project/sglang/pull/40554) [Fix] Add gigachat35 to the tool-call and reasoning parser name lists
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40366](https://github.com/sgl-project/sglang/pull/40366) [sgl-router] refactor - cache-aware policy
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40526](https://github.com/sgl-project/sglang/pull/40526) Clean up startup logging and streamline log audits
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39955](https://github.com/sgl-project/sglang/pull/39955) [ci][xpu] Re-seed the wan2_1_t2v_1.3b perf baseline on Arc Pro B60
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40448](https://github.com/sgl-project/sglang/pull/40448) [Feature] Xiaomi MiMo-V2.6/MiMo-V2.6-Pro day0 support
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#33649](https://github.com/sgl-project/sglang/pull/33649) Update to the cookbook for XPU-supported models
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38932](https://github.com/sgl-project/sglang/pull/38932) fix(modelopt): dispatch NVFP4 MoE on the cached backend, not the live global
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40472](https://github.com/sgl-project/sglang/pull/40472) [diffusion] keep Qwen-Image 2.1 prefix KV per layer under Cache-DiT
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39206](https://github.com/sgl-project/sglang/pull/39206) [Diffusion] Guard E2E/loading latency with runner-aware baselines
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40502](https://github.com/sgl-project/sglang/pull/40502) [Fix] Raise on undelivered embeddings in `send_with_url`, fix broken tests
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40379](https://github.com/sgl-project/sglang/pull/40379) [sgl-router] refactor - session-aware policy
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40272](https://github.com/sgl-project/sglang/pull/40272) [sgl-router] refactor - move policy-required states under src/state
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40496](https://github.com/sgl-project/sglang/pull/40496) [CI] Give the kernel lane a 5090 suite and move kernel-only tests off the general lane
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39200](https://github.com/sgl-project/sglang/pull/39200) [Perf] Fuse the glm5_next mHC attn->MLP boundary
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#40487](https://github.com/sgl-project/sglang/pull/40487) [Diffusion] Add verified DGX Spark recipe for Qwen-Image 2.1

---
*Auto-generated by npu_pr_monitor.py*