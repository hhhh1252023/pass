# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-27 09:44 UTC
**本次检查已合入 PR 数**: 40
**涉及 NPU**: 11 | **无关**: 29 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#35349](https://github.com/sgl-project/sglang/pull/35349) [VLM] Size the multimodal preprocessing pool by where preprocessing runs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#35947](https://github.com/sgl-project/sglang/pull/35947) Publish gated DSV4 DFLASH-family target-prefill read completion
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#36636](https://github.com/sgl-project/sglang/pull/36636) [AMD][CI] Add targeted Mori test labels
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36100](https://github.com/sgl-project/sglang/pull/36100) [ci] xpu: trigger pr-test-xpu on multimodal_gen changes
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#29708](https://github.com/sgl-project/sglang/pull/29708) [KDA-Pilot] Add LTX2 QKNorm split-RoPE CUDA fast path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#24689](https://github.com/sgl-project/sglang/pull/24689) [NPU] Add GitHub test summary and deduplicate test code. Part 2
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#36396](https://github.com/sgl-project/sglang/pull/36396) [AMD][CI] Add DeepSeek-V4-Flash FP8 accuracy coverage on MI30x
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#35342](https://github.com/sgl-project/sglang/pull/35342) [VLM] Route every multimodal processor through the worker pool's call site
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 34

### [#35222](https://github.com/sgl-project/sglang/pull/35222) [CPU] Enable ERNIE models on CPU
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#33871](https://github.com/sgl-project/sglang/pull/33871) [Performance] Reduce idle DP work in breakable prefill CUDA graphs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#33561](https://github.com/sgl-project/sglang/pull/33561) [Model] Support Ling-3.0-flash (BailingMoeV3) 
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 76

## ✅ 与 NPU 无关的已合入 PR
- [#36586](https://github.com/sgl-project/sglang/pull/36586) [Core] Refactor server argument choices
- [#36605](https://github.com/sgl-project/sglang/pull/36605) [CI] Graceful teardown for the radix_cache server fixtures
- [#36639](https://github.com/sgl-project/sglang/pull/36639) [CI] Fix Q8KV8 sparse prefill test fixture
- [#35275](https://github.com/sgl-project/sglang/pull/35275) [Bug][Spec] fix startup crash and reduce CUDA graph memory usage for speculative adaptive
- [#35944](https://github.com/sgl-project/sglang/pull/35944) Pin scheduler metadata before asynchronous H2D copies
- [#36330](https://github.com/sgl-project/sglang/pull/36330) [AMD] Optimize Qwen3.5 MTP unified attention on gfx950
- [#36198](https://github.com/sgl-project/sglang/pull/36198) [Weight Cache] Enhance test and support EPLB
- [#36413](https://github.com/sgl-project/sglang/pull/36413) [CPU][CI]: fix a few issues that cause XEON CI failures
- [#27392](https://github.com/sgl-project/sglang/pull/27392) [KDA-Pilot] Add B200 diffusion norm-scale-shift CUDA fast path for Qwen-Image
- [#29281](https://github.com/sgl-project/sglang/pull/29281) [KDA-Pilot] Add diffusion causal Conv3D cat-pad CUDA fast path for Cosmos3
- [#36608](https://github.com/sgl-project/sglang/pull/36608) [AMD] Add GLM-5.3-Flash recipes for MI300X, MI325X, and MI355X
- [#36541](https://github.com/sgl-project/sglang/pull/36541) [AMD] Fix int32 seqused_k overflow in aiter draft-extend attention
- [#36588](https://github.com/sgl-project/sglang/pull/36588) [intel_dev branch] Rebase main and add DSv4 xpu opt
- [#35791](https://github.com/sgl-project/sglang/pull/35791) [Radix Cache] Add test-only TreeCore inspector for shared backend tests
- [#36609](https://github.com/sgl-project/sglang/pull/36609) [CI] Grant no-cooldown for UNIDY2002
- [#35416](https://github.com/sgl-project/sglang/pull/35416) docs: sync LMSYS SGLang blog cards
- [#36602](https://github.com/sgl-project/sglang/pull/36602) [CI] Remove GLM-4.1V-9B-Thinking from nightly VLM MMMU eval
- [#7872](https://github.com/sgl-project/sglang/pull/7872) [CI] Add deepep tests to CI
- [#35634](https://github.com/sgl-project/sglang/pull/35634) [Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend 
- [#34296](https://github.com/sgl-project/sglang/pull/34296) [AMD] Use fast exponentials in C4 and C128 ROCm kernels
- [#36309](https://github.com/sgl-project/sglang/pull/36309) [AMD][Bugfix] Skip invalid fused MoE reduction for direct top-1 output
- [#36443](https://github.com/sgl-project/sglang/pull/36443) [CPU] Fix rotary_embedding_cpu fake for in-place layouts
- [#36307](https://github.com/sgl-project/sglang/pull/36307) [AMD][CI] Stabilize PyTorch sampling backend test on ROCm
- [#36584](https://github.com/sgl-project/sglang/pull/36584) Fix BailingMoeV3 reading enable_dp_lm_head off live topology instead of config
- [#36430](https://github.com/sgl-project/sglang/pull/36430) [CPU] Fix truncated KV prefix in intel_amx spec verify
- [#36360](https://github.com/sgl-project/sglang/pull/36360) [Intel XPU] Fix cross-encoder rerank hang on B580 runners
- [#36343](https://github.com/sgl-project/sglang/pull/36343) [AMD] Fall back to CPU tensor for decode retraction on ROCm
- [#35379](https://github.com/sgl-project/sglang/pull/35379) [Spec] Generalize hybrid SWA MTP draft pool routing
- [#36573](https://github.com/sgl-project/sglang/pull/36573) Fix _is_compiling dynamo tracing: import torch instead of sys.modules lookup

---
*Auto-generated by npu_pr_monitor.py*