# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-28 23:52 UTC
**本次检查已合入 PR 数**: 34
**涉及 NPU**: 14 | **无关**: 20 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#29718](https://github.com/sgl-project/sglang/pull/29718) [MoE] Make simulated expert routing support DP>1, and fuse into one triton kernel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#36778](https://github.com/sgl-project/sglang/pull/36778) fix: report the real backend for non-CUDA CI registrations in /rerun-test
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36657](https://github.com/sgl-project/sglang/pull/36657) [Blackwell] Reserve SMs for DeepGEMM MegaMoE grid barriers
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#36892](https://github.com/sgl-project/sglang/pull/36892) [AMD] release rocm10 image for gfx1250 from amd_helios
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36790](https://github.com/sgl-project/sglang/pull/36790) config: the derived parallel widths are computed from the leaves
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#36789](https://github.com/sgl-project/sglang/pull/36789) config: the resolution pipeline moves out of the record
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 30

### [#36583](https://github.com/sgl-project/sglang/pull/36583) Fix KV cache pool sized far too small when weight-loading memory is still referenced
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36433](https://github.com/sgl-project/sglang/pull/36433) [NPU] Update sgl-kernel-npu version
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#36862](https://github.com/sgl-project/sglang/pull/36862) [Fix] Route the Mooncake MoE A2A backend through Kimi K3's EP-A2A / SP-MoE fast path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36849](https://github.com/sgl-project/sglang/pull/36849) cp multi bs 
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#36759](https://github.com/sgl-project/sglang/pull/36759) bugfix for index_fill_ on NPU
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#35290](https://github.com/sgl-project/sglang/pull/35290) [XPU] Lazily import tvm_ffi-dependent all_reduce kernel in minimax_m2
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36603](https://github.com/sgl-project/sglang/pull/36603) fix(kimi-k3): preserve dense ModelSlim MLA weights
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#34690](https://github.com/sgl-project/sglang/pull/34690) [BugFix][VLM] keep Qwen3-VL MoE inference deepstack order
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

## ✅ 与 NPU 无关的已合入 PR
- [#36921](https://github.com/sgl-project/sglang/pull/36921) fix: KT's last MoE layer stops deferring experts again
- [#36912](https://github.com/sgl-project/sglang/pull/36912) Use kernel build node for cu134 image
- [#36909](https://github.com/sgl-project/sglang/pull/36909) [mem_cache] Carry `swa_evicted_seqlen` into `SWARadixCache.cache_unfinished_req`
- [#36914](https://github.com/sgl-project/sglang/pull/36914) [Fix] Lazy-import aiter in DSv4 paged_decode to unbreak CPU CI
- [#36754](https://github.com/sgl-project/sglang/pull/36754) [Diffusion] Rollout API: off-loop serialization, spliced msgpack, timing headers, opt-in uint8 video
- [#36094](https://github.com/sgl-project/sglang/pull/36094) [AMD][DSV4] perf: retune decode split-K heuristic for MI355X
- [#36884](https://github.com/sgl-project/sglang/pull/36884) fix(glm): mirror should_use_dp_reduce_scatterv() into the MHC communicator
- [#36638](https://github.com/sgl-project/sglang/pull/36638) Fix KeyError on batch requests whose state is freed before it is read
- [#36738](https://github.com/sgl-project/sglang/pull/36738) [HiCache] Fence load-back behind the forward stream
- [#36382](https://github.com/sgl-project/sglang/pull/36382) [HiCache] Key storage prefetch by the request namespace
- [#36792](https://github.com/sgl-project/sglang/pull/36792) config: the forwarding slots go; the dispatcher calls the family directly
- [#36791](https://github.com/sgl-project/sglang/pull/36791) config: three cache and pool readers take the bags
- [#36673](https://github.com/sgl-project/sglang/pull/36673) [CI] Read subprocess stdout on a background thread to avoid EOF deadlock
- [#36425](https://github.com/sgl-project/sglang/pull/36425) HiCache: avoid unnecessary all-reduce in check_prefetch_progress
- [#36775](https://github.com/sgl-project/sglang/pull/36775) CI: split JIT kernel unit tests into two partitions
- [#36434](https://github.com/sgl-project/sglang/pull/36434) [AMD] Add ROCm 10 (gfx942 / gfx950) release images
- [#36827](https://github.com/sgl-project/sglang/pull/36827) [Docs] Add GLM-5.3 cookbook
- [#35613](https://github.com/sgl-project/sglang/pull/35613) [diffusion] refactor: scope model-specific API parameters
- [#36806](https://github.com/sgl-project/sglang/pull/36806) fix(qsa): route exact SM120 to FlashInfer sparse decode
- [#35677](https://github.com/sgl-project/sglang/pull/35677) fix(cpu): skip GPU JIT MoE top-k on CPU

---
*Auto-generated by npu_pr_monitor.py*