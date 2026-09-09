# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-09 09:07 UTC
**本次检查已合入 PR 数**: 31
**涉及 NPU**: 15 | **无关**: 16 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#38014](https://github.com/sgl-project/sglang/pull/38014) ci(xpu): merge stage-a+b into one job and trim main_package scope
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#38564](https://github.com/sgl-project/sglang/pull/38564) [Fix] Stamp sequence-parallel state on dummy forward batches
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#34820](https://github.com/sgl-project/sglang/pull/34820) Store mamba prefix-cache checkpoints at the configured SSM state dtype
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 17

### [#37303](https://github.com/sgl-project/sglang/pull/37303) [Rust TreeCore] Harden runtime and CI parity
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 22

### [#38591](https://github.com/sgl-project/sglang/pull/38591) [Diffusion] Enable lossless BCG for FLUX.1-dev
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#38530](https://github.com/sgl-project/sglang/pull/38530) [Diffusion] Fuse LongCat Image normalization and modulation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36230](https://github.com/sgl-project/sglang/pull/36230) [CP V1 Deprecation 5/5] Update prefill CP documentation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#38174](https://github.com/sgl-project/sglang/pull/38174) [NPU]support mf device urma and host rdma trans type
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#38182](https://github.com/sgl-project/sglang/pull/38182) [Diffusion] Keep the Wan VAE decoder channels_last and add a Triton NHWC nearest upsample
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#33089](https://github.com/sgl-project/sglang/pull/33089) [NPU] Add sparsity-driven KV offload for DeepSeek DSA on Ascend
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#38506](https://github.com/sgl-project/sglang/pull/38506) [Diffusion] Support mixed INT8 embeddings and Comfy NVFP4 encoders
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#38375](https://github.com/sgl-project/sglang/pull/38375) [Config] Retire get_global_server_args, and clear the deprecated flags that have a replacement
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 100

### [#36557](https://github.com/sgl-project/sglang/pull/36557) MiniMax-M3: Triton split-K router GEMV with in-kernel fixup
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36527](https://github.com/sgl-project/sglang/pull/36527) MiniMax-M3: share the sparse index top-k across layers and reuse the decode top-k buffer
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#36229](https://github.com/sgl-project/sglang/pull/36229) [CP V1 Deprecation 4/5] Canonicalize prefill CP API names
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 33

## ✅ 与 NPU 无关的已合入 PR
- [#38612](https://github.com/sgl-project/sglang/pull/38612) [Kimi-K3] Accept fp32 routing weights in the fused MoE finalize
- [#38617](https://github.com/sgl-project/sglang/pull/38617) docker(xpu): unblock nightly build (setvars.sh + sgl-kernel rename)
- [#38590](https://github.com/sgl-project/sglang/pull/38590) [Attention] Size FlashInfer MLA indptr buffers to the padded max batch
- [#38638](https://github.com/sgl-project/sglang/pull/38638) use private --shm-size instead of --ipc=host to stop /dev/shm leak
- [#36606](https://github.com/sgl-project/sglang/pull/36606) [Diffusion][SenseNova] support SenseNova-U1.5-8B-MoT 
- [#38462](https://github.com/sgl-project/sglang/pull/38462) [mem_cache] skip duplicates host evict via environ
- [#38329](https://github.com/sgl-project/sglang/pull/38329) [AMD][Fix] Fix regression in jit build error with FLUX.2-dev on gfx1250
- [#38535](https://github.com/sgl-project/sglang/pull/38535) [diffusion] Add explicit snapshot-offload component residency
- [#38629](https://github.com/sgl-project/sglang/pull/38629) [diffusion] ci: don't fail the nightly when a perf dump is unreadable
- [#38571](https://github.com/sgl-project/sglang/pull/38571) [AMD][DSV4] Skip the paged SWA page return under the per-request ring
- [#38534](https://github.com/sgl-project/sglang/pull/38534) [Docs] Add measured JoyEcho H200 residency and BCG recipe
- [#27550](https://github.com/sgl-project/sglang/pull/27550) fix(hiradix): wait for extra pool IO
- [#38588](https://github.com/sgl-project/sglang/pull/38588) Cast fp32 routing weights to bf16 in the Kimi-K3 fused finalize
- [#35492](https://github.com/sgl-project/sglang/pull/35492) [CPU] Support Qwen3.8 text+video: adding torchcodec, ffmpeg and removing pin_memory
- [#33366](https://github.com/sgl-project/sglang/pull/33366) [XPU][Diffusion] Enable MiniMax H3 on XPU platforms
- [#38455](https://github.com/sgl-project/sglang/pull/38455) [diffusion] Support MiniMax-H3 Singularity hybrid checkpoints

---
*Auto-generated by npu_pr_monitor.py*