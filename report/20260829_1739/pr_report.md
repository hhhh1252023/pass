# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-29 09:39 UTC
**本次检查已合入 PR 数**: 36
**涉及 NPU**: 10 | **无关**: 26 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#36170](https://github.com/sgl-project/sglang/pull/36170) [NPU] [BugFix] Fix discontinuous input for FIA operator in GLM4.7‑Flash
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36902](https://github.com/sgl-project/sglang/pull/36902) [Diffusion] Delegate recognized quantized components to Transformers
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#34318](https://github.com/sgl-project/sglang/pull/34318) [Kernel] Route large SM90 row/column-scaled FP8 GEMMs to Torch
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#36714](https://github.com/sgl-project/sglang/pull/36714) [AMD][Spec][PD] Enable the PD DSA fused-TopK seed remap on ROCm
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#35021](https://github.com/sgl-project/sglang/pull/35021) [NPU] add causal conv1d for ascend kda backend
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36940](https://github.com/sgl-project/sglang/pull/36940) [NPU] [DOC] udpate supported features on NPU
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36476](https://github.com/sgl-project/sglang/pull/36476) [NPU] [DOC] update npu best practice
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 18

### [#35453](https://github.com/sgl-project/sglang/pull/35453) [Fix] Support LSE on the RadixAttention extra-kwargs graph path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36852](https://github.com/sgl-project/sglang/pull/36852) [ROCm][Bugfix] Use token-level KV indices in the aiter ASM context-prefill gather
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#33576](https://github.com/sgl-project/sglang/pull/33576)  [AMD] Add Work-Centric (Lean) Attention: a persistent-CTA decode kernel for long-context serving
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

## ✅ 与 NPU 无关的已合入 PR
- [#36834](https://github.com/sgl-project/sglang/pull/36834) [HiCache] buffer mode: decide staged-fetch fate against the live tree
- [#34639](https://github.com/sgl-project/sglang/pull/34639) [BugFix] Allow model_loader_extra_config with remote_instance + modelexpress backend
- [#36981](https://github.com/sgl-project/sglang/pull/36981) [CI] Temporarily disable GB300 jobs
- [#36958](https://github.com/sgl-project/sglang/pull/36958) [misc] Keep `req.kv` non-optional and key KV ownership on `req_pool_idx`
- [#34599](https://github.com/sgl-project/sglang/pull/34599) [diffusion] Optimize Pi0.5 inference and bounded graph serving
- [#35739](https://github.com/sgl-project/sglang/pull/35739) [multimodal] Fix NVFP4 diffusion models on sm_120 (RTX PRO 6000 / RTX 50xx)
- [#36863](https://github.com/sgl-project/sglang/pull/36863) [diffusion] Fix image encoder parallel folding proposal
- [#36931](https://github.com/sgl-project/sglang/pull/36931) [diffusion] Honor explicit component offload
- [#36977](https://github.com/sgl-project/sglang/pull/36977) [Cookbook] Run accuracy benchmarks through sgl-eval
- [#36946](https://github.com/sgl-project/sglang/pull/36946) [Docker] Install AI Dynamo nightly
- [#36874](https://github.com/sgl-project/sglang/pull/36874) [Diffusion] Respect component weight overrides for upsamplers
- [#36883](https://github.com/sgl-project/sglang/pull/36883) [Diffusion] Resolve indexed component weight sets
- [#36915](https://github.com/sgl-project/sglang/pull/36915) [AMD] Fix eager metadata for AITER EAGLE draft extend
- [#36963](https://github.com/sgl-project/sglang/pull/36963) [Fix] Fall back to the process-group broadcast for DSA topk when PyNCCL is absent
- [#36920](https://github.com/sgl-project/sglang/pull/36920) Add configurable HTTP/2 connection window
- [#35762](https://github.com/sgl-project/sglang/pull/35762) [PD] Pack DCP1→DCP-N PD KV transfers into dest-contiguous RDMA blocks
- [#35758](https://github.com/sgl-project/sglang/pull/35758) qwen 3.8 rebase
- [#36950](https://github.com/sgl-project/sglang/pull/36950) [Docs] Restore the AIME25 label so GLM-5.3 FP8 and BF16 scores render again
- [#35434](https://github.com/sgl-project/sglang/pull/35434) [CPU] Fix wrongly causal-masked bidirectional attention
- [#36934](https://github.com/sgl-project/sglang/pull/36934) [Fix] Drop the duplicated DSpark draft sample_block call
- [#36929](https://github.com/sgl-project/sglang/pull/36929) Update CUDA 13.4 image to flashinfer 0.6.18rc10, cutedsl 4.8. Fix sgl- wheel unpinning
- [#36828](https://github.com/sgl-project/sglang/pull/36828) [AMD] Update v4 amd cookbook 0828
- [#36704](https://github.com/sgl-project/sglang/pull/36704) Refactor JIT kernel and expert-pack directory layout
- [#36641](https://github.com/sgl-project/sglang/pull/36641) [diffusion] Keep Cosmos3 Nano resident on 96 GB GPUs
- [#36887](https://github.com/sgl-project/sglang/pull/36887) [CI] Slim JIT kernel unit tests
- [#36885](https://github.com/sgl-project/sglang/pull/36885) fix(kda): stop remapping the -1 padding sentinel onto a live mamba slot

---
*Auto-generated by npu_pr_monitor.py*