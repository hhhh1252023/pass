# NPU PR 监控报告 (已合入)
**生成时间**: 2026-07-28 08:04 UTC
**本次检查已合入 PR 数**: 40
**涉及 NPU**: 8 | **无关**: 32 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#25144](https://github.com/sgl-project/sglang/pull/25144) [NPU] Add Ascend NPU support for DeepSeek-V4
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 28

### [#25663](https://github.com/sgl-project/sglang/pull/25663) [MoE Refactor] [NPU] Refactor Ascend MoE implementation to reduce code duplication and align with community design
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 54

### [#29830](https://github.com/sgl-project/sglang/pull/29830) [core/loader] Fix presharded cache-key gaps: moe_dense_tp_size, EPLB with structural signature
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#32409](https://github.com/sgl-project/sglang/pull/32409) [Spec] Hold the grammar bitmask in one `GrammarMask` type across all decode paths
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 16

### [#31346](https://github.com/sgl-project/sglang/pull/31346) fix(dsa): fail fast on fp8_e4m3 KV with tilelang DSA backend on CUDA
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#32205](https://github.com/sgl-project/sglang/pull/32205) Doc/update npu quickstart
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#21511](https://github.com/sgl-project/sglang/pull/21511) [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#30535](https://github.com/sgl-project/sglang/pull/30535) [hicache]: add  mamba_io_kernel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

## ✅ 与 NPU 无关的已合入 PR
- [#32608](https://github.com/sgl-project/sglang/pull/32608) codeowners update
- [#31126](https://github.com/sgl-project/sglang/pull/31126) [Intel XPU] Enable (biased) grouped topk for xpu
- [#32559](https://github.com/sgl-project/sglang/pull/32559) Update mi35x ROCm image to k3-20260727
- [#32547](https://github.com/sgl-project/sglang/pull/32547) docs: point Kimi-K3 references to public branch
- [#32542](https://github.com/sgl-project/sglang/pull/32542) docs(cookbook): add the Kimi-K3 serving cookbook
- [#30157](https://github.com/sgl-project/sglang/pull/30157) Size KV pool after CUDA graph capture (opt-in)
- [#32489](https://github.com/sgl-project/sglang/pull/32489) Add local ZIP uploader for whl releases
- [#26312](https://github.com/sgl-project/sglang/pull/26312) [mtp] add rejection sampling for speculative decoding
- [#32408](https://github.com/sgl-project/sglang/pull/32408) Update audio container test time estimate
- [#32043](https://github.com/sgl-project/sglang/pull/32043) [AMD] GFX1250 ROCm bringup: infra, build, kernels and models (DSV4, DSR1, GPTOSS)
- [#31413](https://github.com/sgl-project/sglang/pull/31413) [Docs] Add Qwen3.6 35B NVFP4 to cookbook
- [#30096](https://github.com/sgl-project/sglang/pull/30096) [DFLASH] Support grammar-constrained decoding in speculative verify
- [#32363](https://github.com/sgl-project/sglang/pull/32363) Add stream label to TTFT metrics
- [#32357](https://github.com/sgl-project/sglang/pull/32357) [Cherry-pick to release/v0.5.16] Fix PyPI release: drop the git-only sgl-eval dep from packaged metadata (#32354)
- [#32350](https://github.com/sgl-project/sglang/pull/32350) [Cherry-pick to release/v0.5.16] Add verbose flag to twine upload command (#32349)
- [#32349](https://github.com/sgl-project/sglang/pull/32349) [chore] Add verbose flag to twine upload command
- [#32346](https://github.com/sgl-project/sglang/pull/32346) [Cherry-pick to release/v0.5.16] Fix stale flashinfer-MLA fallback poisoning spec verify capture (trtllm_mla + tc_piecewise) (#32288)
- [#32324](https://github.com/sgl-project/sglang/pull/32324) [CI] Skip flaky test in CI for disaggregation group
- [#32306](https://github.com/sgl-project/sglang/pull/32306) Add CI permissions for Elastic EP contributor UNIDY2002
- [#32298](https://github.com/sgl-project/sglang/pull/32298) [CI] Fix XPU platform test on machines without the XPU sgl-kernel op
- [#32292](https://github.com/sgl-project/sglang/pull/32292) [Cherry-pick to release/v0.5.16] Fix dynamo recompile limit in allreduce and bf16 gemm (#32239)
- [#32260](https://github.com/sgl-project/sglang/pull/32260) [Cherry-pick to release/v0.5.16] [spec decoding] fix inkling multi layer mtp draft extend cuda graph (#32254)
- [#32259](https://github.com/sgl-project/sglang/pull/32259) [Cherry-pick to release/v0.5.16] Fix nvfp4 online scale with pcg (#32246)
- [#32246](https://github.com/sgl-project/sglang/pull/32246) Fix nvfp4 online scale with pcg
- [#32191](https://github.com/sgl-project/sglang/pull/32191) Rebase to v0.5.15.post1 for internal testing
- [#32193](https://github.com/sgl-project/sglang/pull/32193) [Tiny] Skip sm120 deepgemm test temporarily
- [#25311](https://github.com/sgl-project/sglang/pull/25311) perf(mla): TMA bulk-store set_mla_kv_buffer (up to 12× over baseline)
- [#32174](https://github.com/sgl-project/sglang/pull/32174) [Tiny]Correct runner for testing deepgemm
- [#32142](https://github.com/sgl-project/sglang/pull/32142) [AMD] Update ROCM version from 7.15.0a20260710 to 7.15.0a20260712
- [#32134](https://github.com/sgl-project/sglang/pull/32134) [AMD] Fix GPT-OSS-120b break on gfx1250
- [#32127](https://github.com/sgl-project/sglang/pull/32127) docs: sync LMSYS SGLang blog cards
- [#30832](https://github.com/sgl-project/sglang/pull/30832) Add 'anyOf' schema support for qwen3_coder tool call parser

---
*Auto-generated by npu_pr_monitor.py*