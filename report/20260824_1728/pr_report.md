# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-24 09:28 UTC
**本次检查已合入 PR 数**: 56
**涉及 NPU**: 15 | **无关**: 41 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#35686](https://github.com/sgl-project/sglang/pull/35686) [AMD][CI] Name the ROCm Image That Actually Ran in AMD Job Names
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#34715](https://github.com/sgl-project/sglang/pull/34715) [bugfix] [NPU] fix transpose batch matmul K*B exceed 65536.
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#36092](https://github.com/sgl-project/sglang/pull/36092) Npu single node test timeout config
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#34915](https://github.com/sgl-project/sglang/pull/34915) [MoE] Gather the cutlass MoE activation and its scales in one launch
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#33057](https://github.com/sgl-project/sglang/pull/33057) fix(xpu): enable compressed-tensors FP8 W8A8 on XPU (RedHatAI FP8-dynamic models)
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#33354](https://github.com/sgl-project/sglang/pull/33354) [XPU] Use a fused GDN kernel from sgl-kernel for Qwen3.5
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

### [#35180](https://github.com/sgl-project/sglang/pull/35180) [Quantization] Share bounded post-load device staging
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#35832](https://github.com/sgl-project/sglang/pull/35832) [diffusion] fix a refit KeyError on mapped weights, and stop claiming strides the reload discards
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#35915](https://github.com/sgl-project/sglang/pull/35915) [OpenAI] Drop empty assistant turns for mistral_common tokenizers
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36075](https://github.com/sgl-project/sglang/pull/36075) [Diffusion] Project MiniMax H3 LoRA onto pruned AdaLN
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#36123](https://github.com/sgl-project/sglang/pull/36123) [NPU] [DOC] Polish English wording in NPU docs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#36085](https://github.com/sgl-project/sglang/pull/36085) [Diffusion] Support VAE weight-file overrides
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#36063](https://github.com/sgl-project/sglang/pull/36063) [Diffusion] Reuse SRT quantization contracts and MXFP8 kernels
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 16

### [#35981](https://github.com/sgl-project/sglang/pull/35981) [diffusion] Flatten Wan VAE RMSNorm row addressing
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#35383](https://github.com/sgl-project/sglang/pull/35383) [AMD][CI] Add the Qwen3.8 MXFP4 MI35x nightly
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

## ✅ 与 NPU 无关的已合入 PR
- [#35072](https://github.com/sgl-project/sglang/pull/35072) [Intel XPU] support prefill only models for xpu
- [#35961](https://github.com/sgl-project/sglang/pull/35961) [diffusion] Reuse SANA fast paths in SANA-Video BCG
- [#36070](https://github.com/sgl-project/sglang/pull/36070) [Diffusion] Load pruned MiniMax H3 components natively
- [#36052](https://github.com/sgl-project/sglang/pull/36052) [Diffusion] Load self-describing Quanto INT8 encoders
- [#36084](https://github.com/sgl-project/sglang/pull/36084) [Diffusion] Add per-component quantization overrides
- [#35775](https://github.com/sgl-project/sglang/pull/35775) [GDN] remove XPU path of causal_conv1d_fn and causal_conv1d_update
- [#34462](https://github.com/sgl-project/sglang/pull/34462) [Triton] Bound the sliding-window extend-attention KV loop: -86.6% on SWA layers, -9.4% prefill GPU, bit-identical
- [#36124](https://github.com/sgl-project/sglang/pull/36124) [AMD] Quark shared-experts gate: recognise a trailing MTP layer
- [#36150](https://github.com/sgl-project/sglang/pull/36150) fix(mini-lb): forward the flush_cache timeout param to workers
- [#34461](https://github.com/sgl-project/sglang/pull/34461) [ROCm] Extend the gfx950 extend-attention tile to head_dim <= 128: -43% kernel, -14% TTFT, bit-identical
- [#36110](https://github.com/sgl-project/sglang/pull/36110) [sglang-miles] Add SGLANG_DISABLE_MULTIMEM_AG to force the NCCL all-gather path
- [#36062](https://github.com/sgl-project/sglang/pull/36062) [diffusion] cache LoRA-merged weights in files the page cache can hold
- [#36024](https://github.com/sgl-project/sglang/pull/36024) [diffusion] Speed up LingBot high-quality VAE decode
- [#36019](https://github.com/sgl-project/sglang/pull/36019) [diffusion] Honor XDG cache for model overlays
- [#36149](https://github.com/sgl-project/sglang/pull/36149) fix(xpu): read enable_deterministic_inference from the config bag
- [#36146](https://github.com/sgl-project/sglang/pull/36146) xeon ci fail fast strategy change
- [#36009](https://github.com/sgl-project/sglang/pull/36009) [diffusion] Fix Hunyuan QKV pack indexing at production video shapes
- [#36016](https://github.com/sgl-project/sglang/pull/36016) [diffusion] Refresh quality and BCG benchmark skills
- [#36086](https://github.com/sgl-project/sglang/pull/36086) [Diffusion] Add plain component weight overrides
- [#36037](https://github.com/sgl-project/sglang/pull/36037) [diffusion] feat: support loading mixed w4a8 text encoders
- [#36012](https://github.com/sgl-project/sglang/pull/36012) [diffusion] Default Hunyuan VAE to tiled decode
- [#36053](https://github.com/sgl-project/sglang/pull/36053) chore: move cuda_vmm_utils.py under srt/utils/
- [#33323](https://github.com/sgl-project/sglang/pull/33323) [Intel XPU] Add xpu pass for biased_topk and hash_topk
- [#32856](https://github.com/sgl-project/sglang/pull/32856) [CPU] Fix NUMA/core binding for DP ranks
- [#35454](https://github.com/sgl-project/sglang/pull/35454) [Fix] Harden FlashAttention CUDA graph metadata bounds
- [#35995](https://github.com/sgl-project/sglang/pull/35995) [diffusion] Fuse LongCat-Image QKNorm and interleaved RoPE
- [#35993](https://github.com/sgl-project/sglang/pull/35993) [diffusion] Keep LongLive2 components resident on large GPUs
- [#33840](https://github.com/sgl-project/sglang/pull/33840) [XPU] Support softmax_lse in sgl_kernel::fwd API
- [#36082](https://github.com/sgl-project/sglang/pull/36082) [Diffusion] Infer LoRA alpha from safetensors metadata
- [#36068](https://github.com/sgl-project/sglang/pull/36068) [Diffusion] Reuse SRT AutoRound for quantized DiTs
- [#36057](https://github.com/sgl-project/sglang/pull/36057) [Diffusion] Fetch metadata beside nested LoRA weights
- [#36027](https://github.com/sgl-project/sglang/pull/36027) [diffusion] optimization: reuse minimax h3 prompt refinement across outputs
- [#33684](https://github.com/sgl-project/sglang/pull/33684) [Weight Cache] Support static DP/EP layouts
- [#35227](https://github.com/sgl-project/sglang/pull/35227) Register CPU CI for 17 e2e tests and partition xeon base-c suite
- [#36036](https://github.com/sgl-project/sglang/pull/36036) [Diffusion] Load serialized Comfy W4A8 checkpoints
- [#35506](https://github.com/sgl-project/sglang/pull/35506) [CPU] Add graph register for fused_sigmoid_mul_cpu, fused_qk_gemma_rmsnorm
- [#36078](https://github.com/sgl-project/sglang/pull/36078) [Diffusion] Add composable component weight path CLI
- [#35669](https://github.com/sgl-project/sglang/pull/35669) [CPU] Add check for fused_input_proj in TP=4
- [#36000](https://github.com/sgl-project/sglang/pull/36000) [diffusion] Keep Cosmos3 Nano resident on high-memory GPUs
- [#36008](https://github.com/sgl-project/sglang/pull/36008) [diffusion] Reject unsafe quality=high BCG replay
- [#36005](https://github.com/sgl-project/sglang/pull/36005) [Mamba] fix mamba index h unexpected assertion for dcp

---
*Auto-generated by npu_pr_monitor.py*