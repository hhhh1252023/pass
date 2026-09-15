# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-15 09:11 UTC
**本次检查已合入 PR 数**: 59
**涉及 NPU**: 26 | **无关**: 32 | **不确定**: 1

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#39544](https://github.com/sgl-project/sglang/pull/39544) [misc] Trim redundant variants from the 8-gpu-h20 disaggregation test suite
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#38176](https://github.com/sgl-project/sglang/pull/38176) keeping router GEMM in fp32 for deterministic inference (DeepSeek V3/V4)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#39370](https://github.com/sgl-project/sglang/pull/39370) [DSV4.1] Combine DSpark decode and prefill kernel optimizations
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 58

### [#39322](https://github.com/sgl-project/sglang/pull/39322) [Router] Extract worker selection into policies::selection (no behavior change)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#39423](https://github.com/sgl-project/sglang/pull/39423) [NPU][Bugfix] Disable pinned memory to fix DeepSeek-V2 DP-attention hang
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#39555](https://github.com/sgl-project/sglang/pull/39555) [NPU] [DOC] delete unsupported models in npu docs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39516](https://github.com/sgl-project/sglang/pull/39516) [Fix] HiCache startup ImportError on the pinned kernel wheel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#39366](https://github.com/sgl-project/sglang/pull/39366) fix: stop shadowing the DSpark shared-experts fusion guard
- **检测方式**: 关键词初筛 + AI确认
- **理由**: 删除遮蔽方法后恢复了含 NPU 分支的 shared_experts_fusion_disable_reason，涉及 NPU 逻辑修复。
- **文件数**: 1

### [#39278](https://github.com/sgl-project/sglang/pull/39278) [Fix][Qwen-VL] Normalize <image> sentinel on artifact fast path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#37482](https://github.com/sgl-project/sglang/pull/37482) feat(agent sessions): attribute stored KV cache blocks to sessions
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 22

### [#39108](https://github.com/sgl-project/sglang/pull/39108) [Router] Honor KV-event storage tiers in the cache-aware tree (1/4)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#39219](https://github.com/sgl-project/sglang/pull/39219) [Fix] Don't write conv state from the fused KDA verify kernel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#39389](https://github.com/sgl-project/sglang/pull/39389) [NPU] [DOC] Rename NPU hardware to Ascend A2/A3 Series product
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 66

### [#38833](https://github.com/sgl-project/sglang/pull/38833) [NPU][CI] Add CANN 9.1.0 and Ascend a5 nightly suites
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#39047](https://github.com/sgl-project/sglang/pull/39047) [NPU] Remove temperature/top_p from Qwen3.5-397B-A17B perf test
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39403](https://github.com/sgl-project/sglang/pull/39403) [NPU][CI] Fix sglang.test.ascend import failure in multi-node e2e pods
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#38687](https://github.com/sgl-project/sglang/pull/38687) [Kernel] Add OOT dispatch for clamp position
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#39373](https://github.com/sgl-project/sglang/pull/39373) [Diffusion] docs: give the RTX 5090 its own H3 recipe, measured on a physical desktop
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#35233](https://github.com/sgl-project/sglang/pull/35233) [AMD] Fix registered HiCache host pointer aliases
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 15

### [#39318](https://github.com/sgl-project/sglang/pull/39318) Scope prefetch cache state to the request attempt
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 21

### [#39420](https://github.com/sgl-project/sglang/pull/39420) [DSV4.1] Record side-stream work right before its join to keep CUDA-graph replay on one stream
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#38941](https://github.com/sgl-project/sglang/pull/38941) [Fix] Merge adjacent KV-row frees so a mid-page split under DCP cannot double-free
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#38409](https://github.com/sgl-project/sglang/pull/38409) [Fix] Wait for PDL before reading DeepSeek V4 K cache locations
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39353](https://github.com/sgl-project/sglang/pull/39353) [NPU] Fix device mismatch in SWA mask for DSpark verify graph capture
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#37413](https://github.com/sgl-project/sglang/pull/37413) [AMD][DSV4] feat: enable fp8 two-pool unified_kv on gfx950
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 21

### [#39396](https://github.com/sgl-project/sglang/pull/39396) [docs] DeepSeek-V4: MI355X PD disaggregation recipes for all three strategies
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

## ❓ 不确定是否涉及 NPU 的 PR

### [#39332](https://github.com/sgl-project/sglang/pull/39332) [PD][LoRA] Gate decode admission on adapter slots
- **理由**: AI 调用失败: 500 Server Error: Internal Error for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#38890](https://github.com/sgl-project/sglang/pull/38890) Support non-strict GLM47 tool calls with EBNF constraints
- [#39474](https://github.com/sgl-project/sglang/pull/39474) [qwen 3.8 next] reuse old cuda stream instead of endlessly creating streams
- [#38630](https://github.com/sgl-project/sglang/pull/38630) [profiler] Label draft-runner steps DRAFT and target verify VERIFY in step spans
- [#39553](https://github.com/sgl-project/sglang/pull/39553) [CI] Run test_unified_radix_cache_kl_dcp on cutedsl_mla with bf16 KV cache
- [#39545](https://github.com/sgl-project/sglang/pull/39545) [CI] Wait for a killed test server's GPU memory before the next launch
- [#39534](https://github.com/sgl-project/sglang/pull/39534) [Fix] Aggregate all IPC weight update responses
- [#37748](https://github.com/sgl-project/sglang/pull/37748) [CPU] Implement fused QK Norm and RoPE kernels
- [#39487](https://github.com/sgl-project/sglang/pull/39487) [Fix][DCP] Localize widened KV ids in MLA retraction CPU backup/restore
- [#39374](https://github.com/sgl-project/sglang/pull/39374) [Kimi K3] Optimization stack for 2xB300 EP16: DSPARK-DeepEP, SiTU, TGV, DCP fi_a2a, SP/hook defaults
- [#39061](https://github.com/sgl-project/sglang/pull/39061) Fix MUSA detection in compiled prefill path
- [#39227](https://github.com/sgl-project/sglang/pull/39227) Force reasoning mode for GLM-5.3 chat templates
- [#39148](https://github.com/sgl-project/sglang/pull/39148) [MM] Add flag to force Kimi image preprocessing onto CPU
- [#39368](https://github.com/sgl-project/sglang/pull/39368) [CI][PD] Skip the flaky decode HiCache file-backend disaggregation test
- [#39223](https://github.com/sgl-project/sglang/pull/39223) Fix MegaMoE buffer allocation and caching for effective SM budgets
- [#36821](https://github.com/sgl-project/sglang/pull/36821) [KDA] Support ReplaySSM ring-write in the fused chain-verify kernel
- [#39457](https://github.com/sgl-project/sglang/pull/39457) [sgl-router] Prepare dynamo-render dependencies
- [#38097](https://github.com/sgl-project/sglang/pull/38097) [HiCache] Remove duplicate benchmark result fields
- [#38939](https://github.com/sgl-project/sglang/pull/38939) [Rust] Use Dynamo native renderers when chat templates are missing
- [#39284](https://github.com/sgl-project/sglang/pull/39284) [Benchmark] Add an opt-out for the token-capacity check
- [#39237](https://github.com/sgl-project/sglang/pull/39237) Fix /model_info serialization when a config value is a class
- [#34981](https://github.com/sgl-project/sglang/pull/34981) [model-loader] Split weight loading from postprocessing
- [#39347](https://github.com/sgl-project/sglang/pull/39347) Allow CUDA VMM feature transport with the Rust frontend
- [#36625](https://github.com/sgl-project/sglang/pull/36625) [Logging] Downgrade missing TokenizerManager request state log to warning
- [#39371](https://github.com/sgl-project/sglang/pull/39371) bumping sgl-deep-gemm to 0.2.0
- [#39483](https://github.com/sgl-project/sglang/pull/39483) [dLLM] Add rwang5203 as code owner and grant CI permissions
- [#39446](https://github.com/sgl-project/sglang/pull/39446) Reland fix(qsa): clamp the compress gather to the rows (#38346)
- [#38554](https://github.com/sgl-project/sglang/pull/38554) [Spec] Allow speculative workers to stage prefill shared reads
- [#39445](https://github.com/sgl-project/sglang/pull/39445) [DSV4.1] Multi-stream prepare for ratio-1/2 layers, fused ratio-1 verify compression, PDL on _q_rope_store
- [#39357](https://github.com/sgl-project/sglang/pull/39357) [PD] Preserve the prefill rank during rebootstrap
- [#39414](https://github.com/sgl-project/sglang/pull/39414) [DSV4.1] NVLink collectives, and the DSpark draft head's vocab gather on them
- [#39406](https://github.com/sgl-project/sglang/pull/39406) [AMD] GLM-5.2 MI355X MXFP4: bump image to 20260913, enable TOPK_V2
- [#39405](https://github.com/sgl-project/sglang/pull/39405) [misc] Revert #38346, #33426, #39061 and #39219

---
*Auto-generated by npu_pr_monitor.py*