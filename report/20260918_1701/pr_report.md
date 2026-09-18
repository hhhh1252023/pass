# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-18 09:00 UTC
**本次检查已合入 PR 数**: 34
**涉及 NPU**: 13 | **无关**: 21 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#39881](https://github.com/sgl-project/sglang/pull/39881) [NPU] Fuse MXFP4 W4A8 MoE gmm1 + swiglu + requant into one kernel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#39589](https://github.com/sgl-project/sglang/pull/39589) [NPU] support kimi k3 on A5 and improve performance
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 27

### [#39463](https://github.com/sgl-project/sglang/pull/39463) [Router] Derive error status from a failure class; preserve the worker's status (1/3)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#37969](https://github.com/sgl-project/sglang/pull/37969) [Runtime] Let out-of-tree platforms provide full graph backends
- **检测方式**: 核心路径 + AI确认
- **理由**: 修改平台接口与后端解析逻辑，NPU 作为 out-of-tree 平台可继承该 hook，间接影响其图后端选择。
- **文件数**: 3

### [#39167](https://github.com/sgl-project/sglang/pull/39167) [Router] Shard the cache-aware KV tree by chain root
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#39415](https://github.com/sgl-project/sglang/pull/39415) [NPU] Adapt hicache for K3 hybrid models
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 11

### [#40058](https://github.com/sgl-project/sglang/pull/40058) [NPU] [DOC] delete unsupported api --disable-hybrid-swa-memory for npu
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39919](https://github.com/sgl-project/sglang/pull/39919) [NPU] Avoid repeated BF16 wo_a weight transposes in DeepSeek-V4 decode
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#39953](https://github.com/sgl-project/sglang/pull/39953) Update NPU tag to post4 version and related scripts
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#38780](https://github.com/sgl-project/sglang/pull/38780) [Bug] Guard FlashInfer CUTLASS MoE against 0-token inputs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#34776](https://github.com/sgl-project/sglang/pull/34776) [Fix] Guard conditional top-logprob keys in the completions echo path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#39915](https://github.com/sgl-project/sglang/pull/39915) [gRPC] Stream engine state changes
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#35204](https://github.com/sgl-project/sglang/pull/35204) [Scheduler] Align `RadixCache` no-insert cleanup with `kv_len_to_handle`
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

## ✅ 与 NPU 无关的已合入 PR
- [#40147](https://github.com/sgl-project/sglang/pull/40147) [CI] Add a unified-memory rerun test group
- [#35573](https://github.com/sgl-project/sglang/pull/35573) [ROCm][diffusion] Enable fused qk norm and rope on ROCm
- [#32963](https://github.com/sgl-project/sglang/pull/32963) [NVIDIA][comm] Merge EP+MoE-TP post-experts all-reduces into one _TP reduction
- [#40137](https://github.com/sgl-project/sglang/pull/40137) [Cherry-pick to release/v0.5.20] [gRPC] Stream engine state changes (#39915)
- [#40105](https://github.com/sgl-project/sglang/pull/40105) [MoE] Disable FlashInfer fused finalize by default for numerical accuracy
- [#40121](https://github.com/sgl-project/sglang/pull/40121) Update ci permission
- [#39364](https://github.com/sgl-project/sglang/pull/39364) [DSV4] fix: keep the TileLang JIT cache under SGLANG_CACHE_DIR
- [#39419](https://github.com/sgl-project/sglang/pull/39419) Verify the Ling-3.0-flash-VL FP4 lane on H200 and disable shared-expert fusion in quant recipes
- [#39482](https://github.com/sgl-project/sglang/pull/39482) [Bugfix] Include SM121 in DeepGEMM packed-scale selection
- [#35260](https://github.com/sgl-project/sglang/pull/35260) Fix int4 MoE tuner config filename
- [#40098](https://github.com/sgl-project/sglang/pull/40098) [profiler] Ignore PREBUILT batches in profile-by-stage
- [#19084](https://github.com/sgl-project/sglang/pull/19084) [diffusion] add  /metrics support
- [#40024](https://github.com/sgl-project/sglang/pull/40024) [Scheduler] Add shortest-prefill-first scheduling
- [#40055](https://github.com/sgl-project/sglang/pull/40055) [CI] Check B200 NUMA mapping against sysfs numa_node
- [#39680](https://github.com/sgl-project/sglang/pull/39680) [Kernel] Coalesce the KDA CuTe DSL decode state transpose: ~3x faster, bit-identical
- [#35798](https://github.com/sgl-project/sglang/pull/35798) [Spec] Fix CDF boundary handling in `TreeSpeculativeSamplingTargetOnly`
- [#39690](https://github.com/sgl-project/sglang/pull/39690) [CPU] Avoid prefill CP predicates during decode graph capture
- [#40033](https://github.com/sgl-project/sglang/pull/40033) [Kernel] Move CUDA and ROCm speculative kernels to JIT
- [#40028](https://github.com/sgl-project/sglang/pull/40028) [sglang-miles] Compare MXFP4 Marlin experts in dequantized space in the weight checker
- [#39966](https://github.com/sgl-project/sglang/pull/39966) [Test] Consolidate kernel tests under plural kernels tree
- [#40036](https://github.com/sgl-project/sglang/pull/40036) [Docs] GLM-5.3-Flash cookbook: temporarily remove the DCP option

---
*Auto-generated by npu_pr_monitor.py*