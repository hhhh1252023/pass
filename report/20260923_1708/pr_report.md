# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-23 09:08 UTC
**本次检查已合入 PR 数**: 31
**涉及 NPU**: 12 | **无关**: 6 | **不确定**: 13

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#40879](https://github.com/sgl-project/sglang/pull/40879) [AMD] Drop the unreachable vLLM fallback from ROCm FP8 activation quant
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#33723](https://github.com/sgl-project/sglang/pull/33723) [3/N] elastic-ep: Recapture decode CUDA graphs after scale-up
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 18

### [#35958](https://github.com/sgl-project/sglang/pull/35958) [npu] decoding procedure optimization on qwen3.5/3.6
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#40813](https://github.com/sgl-project/sglang/pull/40813) [AMD][DI][CI] Use a node-local model cache on the SPUR cluster
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40599](https://github.com/sgl-project/sglang/pull/40599) [Diffusion] Add a permanent lifetime for layerwise resident layers
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 10

### [#40568](https://github.com/sgl-project/sglang/pull/40568) [Diffusion] Support MiniMax-H3 PDD(Parallel Decoding Distillation) inference
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#40760](https://github.com/sgl-project/sglang/pull/40760) fix: partition selective CI reruns into matrix jobs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#40824](https://github.com/sgl-project/sglang/pull/40824) [NPU]fix ci hicache oom
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40766](https://github.com/sgl-project/sglang/pull/40766) [sgl-router] Launch reorg routing with existing policy options
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#37704](https://github.com/sgl-project/sglang/pull/37704) [sglang-miles] Kimi K3 colocated RL: LoRA fixes and in-place MXFP4 Marlin reload
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 21

### [#40431](https://github.com/sgl-project/sglang/pull/40431) [dsv4.1]Optimize FP4 indexer by skipping invisible tiles
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40672](https://github.com/sgl-project/sglang/pull/40672) [Fix] Decide the MoE padded-row bound from the layer scatter mode
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 21

## ❓ 不确定是否涉及 NPU 的 PR

### [#39804](https://github.com/sgl-project/sglang/pull/39804) [AMD] Fix DeepSeek-V4 accuracy by not passing num_token_non_padded to MoE topk
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39790](https://github.com/sgl-project/sglang/pull/39790) [ROCm] feat: enable aiter allreduce fusion for GLM models
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39538](https://github.com/sgl-project/sglang/pull/39538) [CPU] Add fused_sigmod_mul_cpu operators to the Meta Muse Glimmer model.
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#31362](https://github.com/sgl-project/sglang/pull/31362) Speculative Decoding with NGRAM support for XPU
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40466](https://github.com/sgl-project/sglang/pull/40466) [deepep_v2] support GLM-5.3-Flash (Glm5NextForConditionalGeneration)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40743](https://github.com/sgl-project/sglang/pull/40743) [Hisparse] fix: account for MiniMax HiSparse full-pool memory
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40468](https://github.com/sgl-project/sglang/pull/40468) [Fix] Keep Inkling automatic tool grammar active across the response
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40792](https://github.com/sgl-project/sglang/pull/40792) Fix NIXL transfer of MXFP8 KV block scales
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40312](https://github.com/sgl-project/sglang/pull/40312) [HiCache] fix: bound the controller reset join so a stalled storage thread cannot hang the scheduler
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40818](https://github.com/sgl-project/sglang/pull/40818) [chore] point agents at the cookbook before test configs
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#34061](https://github.com/sgl-project/sglang/pull/34061) [dLLM] Support DiffusionGemma serving
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40795](https://github.com/sgl-project/sglang/pull/40795) [misc] Remove deprecated endpoints, env vars and aliases past two releases
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40787](https://github.com/sgl-project/sglang/pull/40787) [HiCache] Remove the unused HiRadixCache
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#40831](https://github.com/sgl-project/sglang/pull/40831) [HiCache] ci: add HiCache and unified radix rerun group
- [#40862](https://github.com/sgl-project/sglang/pull/40862) [CI] Update GLM-5.3-Flash H200/B200 test args
- [#39781](https://github.com/sgl-project/sglang/pull/39781) [Intel GPU] Add DeepSeek-V2-Lite-Chat-FP8 gsm8k e2e accuracy nightly test on XPU
- [#40844](https://github.com/sgl-project/sglang/pull/40844) [AMD] Add diffusion (Wan2.2) extras to gfx1151 Docker image
- [#40749](https://github.com/sgl-project/sglang/pull/40749) [HiSparse] ci: add cross-directory rerun test group
- [#40791](https://github.com/sgl-project/sglang/pull/40791) ci: stop Runner Utilization Report from draining the shared API quota

---
*Auto-generated by npu_pr_monitor.py*