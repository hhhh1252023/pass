# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-21 23:12 UTC
**本次检查已合入 PR 数**: 42
**涉及 NPU**: 13 | **无关**: 4 | **不确定**: 25

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#40499](https://github.com/sgl-project/sglang/pull/40499) [Spec][PP] Launch extend microbatches before the spec output exchange
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#40617](https://github.com/sgl-project/sglang/pull/40617) [Test] Anchor `basic_perf` thresholds to each metric's measured spread
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#40345](https://github.com/sgl-project/sglang/pull/40345) Bringing the parallel runtime up becomes a phase, not a side effect
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#40343](https://github.com/sgl-project/sglang/pull/40343) Retire the per-runner parallel record
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 79

### [#40342](https://github.com/sgl-project/sglang/pull/40342) Deprecate the parallel getters the context answers, and ratchet them shut
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 44

### [#40340](https://github.com/sgl-project/sglang/pull/40340) Check the topology identities where the layout is written, and build at the published widths
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 43

### [#40610](https://github.com/sgl-project/sglang/pull/40610) Update DeepSeek-V4 Pro for B200 FP4 agentic PD disaggregation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#39986](https://github.com/sgl-project/sglang/pull/39986) [AMD] Use Triton softmax routing for Qwen3.5 on gfx950
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40256](https://github.com/sgl-project/sglang/pull/40256) Preallocate HiCache MHA staging before post-capture KV sizing
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#37889](https://github.com/sgl-project/sglang/pull/37889) [AMD] Enable GLM DSA prefill top-k to the v2 kernel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#40577](https://github.com/sgl-project/sglang/pull/40577) [Docs][NPU] Add MiMo-V2.5-Pro FP4 DFlash best practice on Ascend NPU
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#40575](https://github.com/sgl-project/sglang/pull/40575) [NPU] [DOC] Add kimi k3 cookbook for 950PR/DT Series
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#40390](https://github.com/sgl-project/sglang/pull/40390) [sgl-router] Add Kimi-K3 rendering with SGLang parity
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 16

## ❓ 不确定是否涉及 NPU 的 PR

### [#39775](https://github.com/sgl-project/sglang/pull/39775) [ROCm] fix: remove extra bf16 -> fp32 cast in jit grouped topk kernel path
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40527](https://github.com/sgl-project/sglang/pull/40527) [CI] Split the CI control labels into four axes and resolve them live
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39339](https://github.com/sgl-project/sglang/pull/39339) [AMD] [GLM-5.3-Flash Day 0] Build the fused DSA k-pool top-k JIT kernel on HIP
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40632](https://github.com/sgl-project/sglang/pull/40632) [Refactor] Clean up parallel runtime comments
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40607](https://github.com/sgl-project/sglang/pull/40607) Fix GLM-5.3 forget-gate shape for nvCUTEDSL verify
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39987](https://github.com/sgl-project/sglang/pull/39987) [AMD] Tune Qwen3.5 TP4 GDN recurrent launch on gfx950
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40620](https://github.com/sgl-project/sglang/pull/40620) [CI] Bump sgl-eval to 0.1.2
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39175](https://github.com/sgl-project/sglang/pull/39175) [Fix] Don't free the multi-CTAs KV counter the decode graphs captured
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40344](https://github.com/sgl-project/sglang/pull/40344) Take the parallel getters off the package's public surface
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40341](https://github.com/sgl-project/sglang/pull/40341) A runner and the objects it builds freeze the placement they describe
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40602](https://github.com/sgl-project/sglang/pull/40602) chore: add NIXL owners and CI access
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40339](https://github.com/sgl-project/sglang/pull/40339) State the draft's whole topology in its scope, and read the rest from the context
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39026](https://github.com/sgl-project/sglang/pull/39026) feat: use XGrammar V4.1 DSML parameter constraints
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40611](https://github.com/sgl-project/sglang/pull/40611) Revert "[Diffusion] migrate the whole _register_configs from registry.py to the model own config file"
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#37762](https://github.com/sgl-project/sglang/pull/37762) [AMD] Fix DeepSeek-R1-MXFP4 accuracy with AITER FP8
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40475](https://github.com/sgl-project/sglang/pull/40475) [Diffusion] migrate the whole _register_configs from registry.py to the model own config file
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40278](https://github.com/sgl-project/sglang/pull/40278) [HiCache] TMA-staged host<->device KV transfer kernel (sm_90+)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40517](https://github.com/sgl-project/sglang/pull/40517) [KDA] Enable ReplaySSM for GLM-5.3 Flash
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#35351](https://github.com/sgl-project/sglang/pull/35351) [mxfp8-kv] Skip writes to the reserved CUDA-graph padding slot
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40391](https://github.com/sgl-project/sglang/pull/40391) [sgl-router] Bound streaming lifetimes and release guards on idle disconnect
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40505](https://github.com/sgl-project/sglang/pull/40505) [Test] Split the serving perf tests by topic into `basic_perf/` and route their thresholds through a kit
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39993](https://github.com/sgl-project/sglang/pull/39993) [Observability] Expose python/rust frontend identity in `/server_info`
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40573](https://github.com/sgl-project/sglang/pull/40573) [docs] Qwen-Image-2.1 cookbook: ComfyUI sections, trimmed examples, and the RTX 5090 DiT-resident recipe (1.42x)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40532](https://github.com/sgl-project/sglang/pull/40532) [sgl-router] Add SGLang-compatible DeepSeek V4.1 Flash rendering
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40530](https://github.com/sgl-project/sglang/pull/40530) [sgl-router] Match DeepSeek V4 rendering to SGLang
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#40603](https://github.com/sgl-project/sglang/pull/40603) [sgl-router] Release cancelled circuit-breaker probes
- [#40570](https://github.com/sgl-project/sglang/pull/40570) [AMD] Enable HiCache for GLM-5.2 MI355X throughput recipe
- [#40622](https://github.com/sgl-project/sglang/pull/40622) Add MiMo-V2.6 cookbook
- [#40618](https://github.com/sgl-project/sglang/pull/40618) Fix lint failure from MXFP8 reserved-slot test location

---
*Auto-generated by npu_pr_monitor.py*