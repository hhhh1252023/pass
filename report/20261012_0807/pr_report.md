# NPU PR 监控报告 (已合入)
**生成时间**: 2026-10-12 00:07 UTC
**本次检查已合入 PR 数**: 22
**涉及 NPU**: 8 | **无关**: 0 | **不确定**: 14

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#43669](https://github.com/sgl-project/sglang/pull/43669) [DSpark] Merge the draft TP sync sites into `dspark-draft`
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#43668](https://github.com/sgl-project/sglang/pull/43668) [DSpark] Run the same markov block in the draft graph tail; drop `SGLANG_DSPARK_FOLDED_SAMPLING`
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 13

### [#43621](https://github.com/sgl-project/sglang/pull/43621) [DSpark] Add `MarkovSampleStep`, one op for a markov draft step's add and token pick
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#43567](https://github.com/sgl-project/sglang/pull/43567) [Unified Radix] Invalidate lazy Mamba checkpoints before in-place tracking
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#43596](https://github.com/sgl-project/sglang/pull/43596) Build Mamba host pools from device buffer infos
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#35721](https://github.com/sgl-project/sglang/pull/35721) [BugFix][DP] Prevent health checks from perturbing user request routing
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#43062](https://github.com/sgl-project/sglang/pull/43062) [Diffusion] Add ModelOpt FP8 checkpoint quantization for Pi0.5
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#27372](https://github.com/sgl-project/sglang/pull/27372) [PD] Fix KV cache corruption on abort by notifying ongoing prefill
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

## ❓ 不确定是否涉及 NPU 的 PR

### [#43623](https://github.com/sgl-project/sglang/pull/43623) [DSpark] Apply `SGLANG_SIMULATE_ACC_LEN` inside the accept instead of switching paths
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43670](https://github.com/sgl-project/sglang/pull/43670) [DSpark] Add `SGLANG_DSPARK_USE_GREEDY_DRAFT`, an argmax draft accepted as one-hot
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43622](https://github.com/sgl-project/sglang/pull/43622) [DSpark] Draft the eager markov block through `MarkovSampleStep`
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43620](https://github.com/sgl-project/sglang/pull/43620) [DSpark] Run every markov head through one step interface
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#42455](https://github.com/sgl-project/sglang/pull/42455) [KV Events] Publish LoRA adapter names and namespaced block hashes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#35292](https://github.com/sgl-project/sglang/pull/35292) fix: default Inkling to fa4 on Hopper
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43550](https://github.com/sgl-project/sglang/pull/43550) [HiCache] Keep storage workers alive on backend exceptions
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43690](https://github.com/sgl-project/sglang/pull/43690) [diffusion] fix: select VSA-H3 for FastH3 when no transformer backend is given
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43593](https://github.com/sgl-project/sglang/pull/43593) Preserve serialized page dtype in NIXL copy fallback
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43595](https://github.com/sgl-project/sglang/pull/43595) Share MLA host capacity initialization
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43386](https://github.com/sgl-project/sglang/pull/43386) [kvevents] fix missing topics in replayed kv events
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43656](https://github.com/sgl-project/sglang/pull/43656) [diffusion] Encode LTX video/audio through the shared bounded MP4 path
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43475](https://github.com/sgl-project/sglang/pull/43475) [Fix] Repair multimodal VMM lease cleanup and source-stream ordering
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#43070](https://github.com/sgl-project/sglang/pull/43070) [DeepSeek-V4.1] Use num_stages=1 for the hc_mix stats partial kernel
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

---
*Auto-generated by npu_pr_monitor.py*