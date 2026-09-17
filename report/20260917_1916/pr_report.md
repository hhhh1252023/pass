# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-17 11:16 UTC
**本次检查已合入 PR 数**: 38
**涉及 NPU**: 13 | **无关**: 25 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#38420](https://github.com/sgl-project/sglang/pull/38420) [NPU]Refactor weight processing and add NPUSwigluLimit activation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#38184](https://github.com/sgl-project/sglang/pull/38184) [AMD][Spec] Enable GDN ReplaySSM target-verify on ROCm
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#17228](https://github.com/sgl-project/sglang/pull/17228) [NPU] make torch_native lora backend a little bit faster
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39382](https://github.com/sgl-project/sglang/pull/39382) [NPU][Diffusion] Optimize SenseNova-U1 batched generation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 11

### [#39666](https://github.com/sgl-project/sglang/pull/39666) dsv4.1: Engram module and request history support
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#39398](https://github.com/sgl-project/sglang/pull/39398) [Benchmark] Limit warmup concurrency in serving benchmark
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39796](https://github.com/sgl-project/sglang/pull/39796) [XPU] Allow the nightly XPU docker build to target a branch or tag
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36825](https://github.com/sgl-project/sglang/pull/36825) [diffusion] Fix the XPU capability gates that broke the Wan2.2 A14B DiT path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 12

### [#39439](https://github.com/sgl-project/sglang/pull/39439) [XPU] weekly simple model enablement 2026/09/14
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 11

### [#39874](https://github.com/sgl-project/sglang/pull/39874) Support LongBench v2 in one-batch server benchmarks
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39668](https://github.com/sgl-project/sglang/pull/39668) dsv4.1: vision tower and image preprocessing
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#39664](https://github.com/sgl-project/sglang/pull/39664) dsv4.1: mHC computation and compensated projections
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#39656](https://github.com/sgl-project/sglang/pull/39656) dsv4.1: RoPE and FP4 packing kernels
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

## ✅ 与 NPU 无关的已合入 PR
- [#39870](https://github.com/sgl-project/sglang/pull/39870) Carry deferred attention operands and reuse multimodal shared memory
- [#39858](https://github.com/sgl-project/sglang/pull/39858) Restrict SafeUnpickler standard-library globals
- [#39513](https://github.com/sgl-project/sglang/pull/39513) [AMD][Bugfix] Fix vattn_asm HIP error 709 under CUDA graph capture on ROCm 10
- [#38983](https://github.com/sgl-project/sglang/pull/38983) [sgl-router] Render chat prompts with dynamo-render
- [#39920](https://github.com/sgl-project/sglang/pull/39920) [Moe] Fix flashinfer_trtllm silently dropping swiglu_limit clamped SwiGLU activation
- [#39884](https://github.com/sgl-project/sglang/pull/39884) [Diffusion] Remove retired auto-residency workload helpers
- [#37474](https://github.com/sgl-project/sglang/pull/37474) Gate mamba extra-buffer predicates on uses_mamba_radix_cache
- [#39551](https://github.com/sgl-project/sglang/pull/39551) [AMD] Multi-arch ROCm support for internal testing
- [#39835](https://github.com/sgl-project/sglang/pull/39835) Re-land dp-attention local control broadcast test
- [#39115](https://github.com/sgl-project/sglang/pull/39115) [Mamba] Fix checkpoint depth for prefixes that end off the radix page
- [#39910](https://github.com/sgl-project/sglang/pull/39910) [AMD][DSV4] Allow moe_a2a_backend='mori' with DSpark + dp attention
- [#37939](https://github.com/sgl-project/sglang/pull/37939) [CPU] Upgrade torch cpu dependencies to version 2.14.
- [#39882](https://github.com/sgl-project/sglang/pull/39882) [Diffusion] Preserve explicit attention backends during autotune
- [#39906](https://github.com/sgl-project/sglang/pull/39906) [CI] Unify basic and speculative sanity accuracy checks with MMLU
- [#39892](https://github.com/sgl-project/sglang/pull/39892) [CI] Fix sanity evaluation and diffusion test suite blockers
- [#39665](https://github.com/sgl-project/sglang/pull/39665) dsv4.1: chat encoding and tool parsing
- [#38504](https://github.com/sgl-project/sglang/pull/38504) [HiCache] Yield idle scheduler so storage workers can drain
- [#36576](https://github.com/sgl-project/sglang/pull/36576) MiniMax-M3: allow shared-experts fusion on ROCm gfx942 and newer
- [#39699](https://github.com/sgl-project/sglang/pull/39699) [HiCache] Keep hybrid transfer layer maps stage-local under PP
- [#39089](https://github.com/sgl-project/sglang/pull/39089) [HiCache] Back up MXFP8 KV scales in the host pool
- [#39124](https://github.com/sgl-project/sglang/pull/39124) fix(kda_prefill): fence shared writes before async proxy reads
- [#39875](https://github.com/sgl-project/sglang/pull/39875) [AMD][bugfix] Fix dsv4 server launch
- [#39657](https://github.com/sgl-project/sglang/pull/39657) dsv4.1: Hopper FP8 matmul kernels and tuning
- [#39869](https://github.com/sgl-project/sglang/pull/39869) [Fix] Allow closed object schemas in Outlines prevalidation
- [#39855](https://github.com/sgl-project/sglang/pull/39855) [CI] Fix missing elfutils headers in CUDA 13.4 DeepGEMM build

---
*Auto-generated by npu_pr_monitor.py*