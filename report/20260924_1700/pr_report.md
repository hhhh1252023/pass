# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-24 09:00 UTC
**本次检查已合入 PR 数**: 45
**涉及 NPU**: 18 | **无关**: 2 | **不确定**: 25

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#40510](https://github.com/sgl-project/sglang/pull/40510) [NPU] Fix DSV4 hard-coding kv dtype
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#40999](https://github.com/sgl-project/sglang/pull/40999) [ci] pr-gate: add generic require-label input and support pull_request_target
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#40905](https://github.com/sgl-project/sglang/pull/40905) [NPU] Remove the LLaDA2.0-mini basic-function test case
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#41026](https://github.com/sgl-project/sglang/pull/41026) [AMD][DI][CI] Say which image the MI355X nightly ran on
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#40777](https://github.com/sgl-project/sglang/pull/40777) [RL] Add RL weight-update sessions and support updating spec draft runners
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 37

### [#41000](https://github.com/sgl-project/sglang/pull/41000) [NPU] [DOC] Remove --enforce-shared-experts-fusion from npu docs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39607](https://github.com/sgl-project/sglang/pull/39607) [NPU] Support batch invariant FIA graphs for deterministic inference
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#28293](https://github.com/sgl-project/sglang/pull/28293) [NPU] Add NPU fallback for fused Triton gating kernels
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#28223](https://github.com/sgl-project/sglang/pull/28223) [NPU] Add MiMo-V2-Flash manual testcases
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#27752](https://github.com/sgl-project/sglang/pull/27752) [NPU][Bugfix] Fix accuracy issue in no-graph with MTP
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#25455](https://github.com/sgl-project/sglang/pull/25455) [NPU] MiMo-V2-Flash Adaptation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#24595](https://github.com/sgl-project/sglang/pull/24595) [NPU] use causal_conv1d_update_v2 for performance
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#23815](https://github.com/sgl-project/sglang/pull/23815) [NPU] Fix DeepEP LL dispatch BF16 flag and skip triton kernel on NPU for Qwen3.5
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#29131](https://github.com/sgl-project/sglang/pull/29131) [NPU] Adapt MiMo-V2.5-W8A8
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#30638](https://github.com/sgl-project/sglang/pull/30638) [NPU][bugfix] Fix post_capture_active TypeError on NPU by adding param to NPUMHATokenToKVPool
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#27923](https://github.com/sgl-project/sglang/pull/27923) Fix MambaPool.clear_slots OOM by replacing expand-based tensor allocation with scalar zeroing
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36549](https://github.com/sgl-project/sglang/pull/36549) MiniMax-M3: allocate the lightning-indexer K cache in fp8 on gfx95
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#40242](https://github.com/sgl-project/sglang/pull/40242) Resolve HF LoRA targets through model-aware normalization
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

## ❓ 不确定是否涉及 NPU 的 PR

### [#40118](https://github.com/sgl-project/sglang/pull/40118) [Experimental] Preserve speculative decoding during prefill across DP ranks
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41048](https://github.com/sgl-project/sglang/pull/41048) [DSV4] Budget the ratio-2 pair state pool in DSV4PoolConfigurator
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41053](https://github.com/sgl-project/sglang/pull/41053) [AMD][DI][CI] Move MI355X disagg nightly to ROCm 10
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41047](https://github.com/sgl-project/sglang/pull/41047) [cherrypick from #40932] [Sampling] Add selected/support sampling logprob modes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40932](https://github.com/sgl-project/sglang/pull/40932) [Sampling] Add selected/support sampling logprob modes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#37751](https://github.com/sgl-project/sglang/pull/37751) [AMD][Diffusion] FlyDSL fused norm kernels on wave32 targets (gfx1250)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40337](https://github.com/sgl-project/sglang/pull/40337) [DSV4] fix: size the C4 state ring by the page it is addressed by
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40204](https://github.com/sgl-project/sglang/pull/40204) [AMD] Small-M MXFP4 fused-MoE kernel for gfx950 (Qwen)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40238](https://github.com/sgl-project/sglang/pull/40238) [PD] Add decode host receive for custom transfer backends
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41024](https://github.com/sgl-project/sglang/pull/41024) [Docs] DeepSeek-V4 MI355X Pro Official PD pairs with DSpark and UMBP
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40710](https://github.com/sgl-project/sglang/pull/40710) [ROCm][DSA] Enable AITER fused FP8 indexer writer
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40796](https://github.com/sgl-project/sglang/pull/40796) [Fix] Use cached prefix lengths for FlashInfer full-attention ragged prefill
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40754](https://github.com/sgl-project/sglang/pull/40754) [AMD] Critical fix enabling Qwen3.8 FP8: restore dropped fused shared-expert weights
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39939](https://github.com/sgl-project/sglang/pull/39939) [Moe] Honor swiglu_limit clamped activation in flashinfer_cutlass runner
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#37213](https://github.com/sgl-project/sglang/pull/37213) [XPU] Qwen3.8-flash-next enablement
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40804](https://github.com/sgl-project/sglang/pull/40804) [RL] Keep DSA cuda-graph state and the graph pool intact across TMS pause/resume
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40983](https://github.com/sgl-project/sglang/pull/40983) [Fix] Derive per-runner hybrid SWA layer ids on ModelLayerInfo instead of mutating ModelConfig
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#34695](https://github.com/sgl-project/sglang/pull/34695) [AMD] Speed up Wan2.2 DiT FP8 attention per-tensor quantization
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40946](https://github.com/sgl-project/sglang/pull/40946) Revert "[AMD] Fix DeepSeek-V4 accuracy by not passing num_token_non_padded to MoE topk"
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38876](https://github.com/sgl-project/sglang/pull/38876) [AMD] Add a Triton packed sparse decode path for QSA on ROCm
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40963](https://github.com/sgl-project/sglang/pull/40963) [mem_cache] Remove unused helpers in mem_cache, storage backends, and metrics
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#29493](https://github.com/sgl-project/sglang/pull/29493) [NPU][Bugfix] Add scoring_func for mimo_v2
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40989](https://github.com/sgl-project/sglang/pull/40989) [Fix] Recover from stale torch extension locks in every `cpp_extension` loader
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#37284](https://github.com/sgl-project/sglang/pull/37284) [RL] Release the weight-checker snapshot once compare passes
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40649](https://github.com/sgl-project/sglang/pull/40649) fix(nccl): disable graph buffer registration when DP attention replays decode graphs
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#40812](https://github.com/sgl-project/sglang/pull/40812) [AMD] Register mem-cache unit tests in PR CI
- [#40969](https://github.com/sgl-project/sglang/pull/40969) [Doc] Add H200 recipes to MiMo-V2.6 cookbook

---
*Auto-generated by npu_pr_monitor.py*