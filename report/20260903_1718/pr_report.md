# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-03 09:18 UTC
**本次检查已合入 PR 数**: 39
**涉及 NPU**: 12 | **无关**: 27 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#37654](https://github.com/sgl-project/sglang/pull/37654) [Model] Add native IFM K2 Horizon serving support
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 34

### [#37655](https://github.com/sgl-project/sglang/pull/37655) docs: add K2 Horizon cookbook recipes and H200 results
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#36384](https://github.com/sgl-project/sglang/pull/36384) [sglang-miles] Streamed LoRA weight updates: register RPC, session scope, LoRA stash
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 16

### [#37572](https://github.com/sgl-project/sglang/pull/37572) Modify KUBE_JOB_NAME to fix the problem of the string being too long
- **检测方式**: 关键词初筛 + AI确认
- **理由**: 修改了NPU专用CI工作流文件中的KUBE_JOB_NAME变量，属于NPU相关CI配置改动。
- **文件数**: 1

### [#37489](https://github.com/sgl-project/sglang/pull/37489) [Fix] Preserve FP32 in SM107 MXFP8 fallback
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36615](https://github.com/sgl-project/sglang/pull/36615) Add SGLANG_CRASH_ON_JIT_COMPILE to forbid on-the-fly JIT compilation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#37469](https://github.com/sgl-project/sglang/pull/37469) [bench] Support real-traffic replay with early-stop-aware steady-state metrics in bench_one_batch_server
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#37550](https://github.com/sgl-project/sglang/pull/37550) Converge the two SWA predicates, and stop conditioning the capture sink on the pool
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#37560](https://github.com/sgl-project/sglang/pull/37560) Fix unified SWA: size a non-owner's v2p by the id space it must address
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#37511](https://github.com/sgl-project/sglang/pull/37511) Size the unified read-table grid from bs, and fuse the allocator's tombstone scatters
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#37146](https://github.com/sgl-project/sglang/pull/37146) [PD] Optimize paged allocator free-list release
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#37505](https://github.com/sgl-project/sglang/pull/37505) [Fix] DP attention: correct the decode->extend prefix off-by-one
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

## ✅ 与 NPU 无关的已合入 PR
- [#36617](https://github.com/sgl-project/sglang/pull/36617) fix test/manual/test_forward_split_prefill.py UT due to many refactors and design changes
- [#37660](https://github.com/sgl-project/sglang/pull/37660) [AMD] Fix FP4 indexer OOR
- [#37723](https://github.com/sgl-project/sglang/pull/37723) [Docs] Update K2 Horizon MoE model names
- [#36349](https://github.com/sgl-project/sglang/pull/36349) [AMD][Diffusion] Migrate FlyDSL fused norm kernels to the v0.3.0 stable API
- [#37118](https://github.com/sgl-project/sglang/pull/37118) [ROCm] Define the DSA head-gate graph helpers on HIP
- [#27862](https://github.com/sgl-project/sglang/pull/27862) Support speculative decoding on CPU
- [#35313](https://github.com/sgl-project/sglang/pull/35313) [CPU] Update base image to Ubuntu 26.04
- [#37675](https://github.com/sgl-project/sglang/pull/37675) [Fix] Broadcast PP dynamic-chunk profiling failures so every rank disables together
- [#37689](https://github.com/sgl-project/sglang/pull/37689) [CI] Accept t64-suffixed apt packages in the install skip check
- [#37320](https://github.com/sgl-project/sglang/pull/37320) [Fix] Alpha-channel images and tool-result media ordering (port of #36507)
- [#37324](https://github.com/sgl-project/sglang/pull/37324) [Perf] Walk the radix tree by offset instead of re-slicing token storage (ported from #36507)
- [#37695](https://github.com/sgl-project/sglang/pull/37695) [chore] Add .git-blame-ignore-revs for the black -> ruff-format reformat (#37210)
- [#33911](https://github.com/sgl-project/sglang/pull/33911) feat(kernels): generalize persistent CuTe JIT cache
- [#37210](https://github.com/sgl-project/sglang/pull/37210) [CI][RFC] Replace black-jupyter with ruff-format
- [#37674](https://github.com/sgl-project/sglang/pull/37674) [misc] Extract PP dynamic chunk sizing into a `DynamicChunkSizer` scheduler component
- [#37193](https://github.com/sgl-project/sglang/pull/37193) Xpu/weekly simple model enablement 2026 08 30
- [#36922](https://github.com/sgl-project/sglang/pull/36922) [chore] harden checkpoint quantization metadata parsing
- [#37329](https://github.com/sgl-project/sglang/pull/37329) Improve CUDA graph and speculative execution output handling
- [#37485](https://github.com/sgl-project/sglang/pull/37485) [CI] Graceful teardown for the PD and HiSparse server fixtures
- [#37487](https://github.com/sgl-project/sglang/pull/37487) Temporarily Remove GLM-5.3 Flash decode CP support
- [#37330](https://github.com/sgl-project/sglang/pull/37330) Reduce tokenizer overhead and offload CUDA VMM publication
- [#37669](https://github.com/sgl-project/sglang/pull/37669) [Fix] Apply the attention-CP broadcast result in PP dynamic-chunk profiling
- [#37576](https://github.com/sgl-project/sglang/pull/37576) [Docs] GLM-5.3-Flash cookbook: drop stale EP caveat, add B300/H100/B200 FP8 speed data
- [#36630](https://github.com/sgl-project/sglang/pull/36630) [Sampling] Capture masks from sampler support
- [#37512](https://github.com/sgl-project/sglang/pull/37512) Build the unified read stream directly, without the page-table rectangle
- [#37672](https://github.com/sgl-project/sglang/pull/37672) [CI] Install lmms-eval from PyPI, drop human-eval install, add clone token fallback
- [#37522](https://github.com/sgl-project/sglang/pull/37522) [CI] Re-enable GB300 jobs

---
*Auto-generated by npu_pr_monitor.py*