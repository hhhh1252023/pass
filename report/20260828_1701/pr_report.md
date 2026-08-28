# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-28 09:01 UTC
**本次检查已合入 PR 数**: 44
**涉及 NPU**: 17 | **无关**: 27 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#36747](https://github.com/sgl-project/sglang/pull/36747) Revert "[NPU] [bugfix] Fix import of ggml_moe_a8_vec and Fix NPU MLA HiCache backup accessing missing data_ptrs"
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36758](https://github.com/sgl-project/sglang/pull/36758) [AMD] Qwen3.5 ASM FMHA chunked-prefill context attention
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36658](https://github.com/sgl-project/sglang/pull/36658) [multimodal_gen] fix: make tail_attn_meta CUDA-graph capturable
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36607](https://github.com/sgl-project/sglang/pull/36607) [AMD] Enable GLM-5.3-Flash on gfx942 and gfx950
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 28

### [#36119](https://github.com/sgl-project/sglang/pull/36119) [AMD][DSV4] perf: MXFP8 MoRI dispatch to match the w4a8 MoE input format
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#36130](https://github.com/sgl-project/sglang/pull/36130) [AMD][DSV4] perf: bound the MoRI receive buffer during decode
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36356](https://github.com/sgl-project/sglang/pull/36356) [AMD] Enable aiter mla asm path through padding attn heads for Kimi K3
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36547](https://github.com/sgl-project/sglang/pull/36547) Fix DeepSeek V4 multistream QKV buffer lifetime
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#36672](https://github.com/sgl-project/sglang/pull/36672) [NPU] Chain PR test jobs and disable two DeepSeek-V4-Flash perf tests
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#34747](https://github.com/sgl-project/sglang/pull/34747) [Cosmos3] Add cosmos3 transfer capability
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 13

### [#36502](https://github.com/sgl-project/sglang/pull/36502) [diffusion] fuse Helios paired transposed RoPE
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#35451](https://github.com/sgl-project/sglang/pull/35451) [Feature] Support PP in full prefill CUDA graphs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#36755](https://github.com/sgl-project/sglang/pull/36755) Fix DFLASH aux hidden-state capture on mHC models
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36529](https://github.com/sgl-project/sglang/pull/36529) [Fix][XPU/ROCm/NPU] Defer sgl_kernel.quantization import in expert_pack
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#36640](https://github.com/sgl-project/sglang/pull/36640) [NPU] [bugfix] Fix import of ggml_moe_a8_vec and Fix NPU MLA HiCache backup accessing missing data_ptrs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#36739](https://github.com/sgl-project/sglang/pull/36739) [misc] Fold the allocator free-group flag into `free_group`
- **检测方式**: 关键词初筛 + AI确认
- **理由**: 修改了NPU后端分配器代码，涉及free_group标志逻辑变更。
- **文件数**: 7

### [#36740](https://github.com/sgl-project/sglang/pull/36740) cookbook: add a Speculative card to the GLM-5.3-Flash playground
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

## ✅ 与 NPU 无关的已合入 PR
- [#35931](https://github.com/sgl-project/sglang/pull/35931) [HiCache] Reject load-back specs that claim nodes pinned by an in-flight load-back
- [#36521](https://github.com/sgl-project/sglang/pull/36521) [diffusion][kernel] avoid 4D scale-shift autotuning
- [#36504](https://github.com/sgl-project/sglang/pull/36504) [diffusion][kernel] support transposed residual-gate add
- [#36227](https://github.com/sgl-project/sglang/pull/36227) [HiCache] Retry L3 storage prefetch after a missed attempt
- [#36386](https://github.com/sgl-project/sglang/pull/36386) [HiCache] Heal the storage existence cache on a hybrid prefetch discard
- [#36667](https://github.com/sgl-project/sglang/pull/36667) chore: make qwen4-main-squashed pre-commit clean
- [#36823](https://github.com/sgl-project/sglang/pull/36823) [Docs] Rename Tencent cookbook page titles to "Hy4 preview" / "Hy3 preview"
- [#34702](https://github.com/sgl-project/sglang/pull/34702) fix(lora): build the MoE LoRA align JIT kernel on ROCm
- [#36379](https://github.com/sgl-project/sglang/pull/36379) fix(lora): build the MoE LoRA align JIT kernel on ROCm
- [#35341](https://github.com/sgl-project/sglang/pull/35341) [AMD][Fix] Qwen3.5: make empty-batch guard tuple-aware on fused AR+quant path
- [#36308](https://github.com/sgl-project/sglang/pull/36308) [AMD][CI] Limit HiCache MGSM eval concurrency on ROCm
- [#36808](https://github.com/sgl-project/sglang/pull/36808) [Cookbook] Hy4-Preview follow-ups: runtime-accurate recipes + released-model info
- [#36804](https://github.com/sgl-project/sglang/pull/36804) [Cookbook] Add the Hy4-Preview model page (Tencent)
- [#36794](https://github.com/sgl-project/sglang/pull/36794) fix(deps): pin compressed-tensors to 0.18.0
- [#36684](https://github.com/sgl-project/sglang/pull/36684) [AMD] Enable deepseek-v4 topk_transform v2 kernel
- [#35628](https://github.com/sgl-project/sglang/pull/35628) [AMD] Increase gfx950 DSA model indexer topk_transform kernel occupancy
- [#36784](https://github.com/sgl-project/sglang/pull/36784) [Docs] Feature GLM-5.3-Flash in the popular-models banner
- [#36626](https://github.com/sgl-project/sglang/pull/36626) [Fix] Resolve tool argument types through top-level anyOf/oneOf/allOf
- [#36772](https://github.com/sgl-project/sglang/pull/36772) fix(qwen4): accept qwen_sparse_attention layer type alias
- [#36726](https://github.com/sgl-project/sglang/pull/36726) [Diffusion] Fix the five unit tests failing on main
- [#36705](https://github.com/sgl-project/sglang/pull/36705) [HiCache] Stop populating host-pool mmaps twice (-13% allocation time)
- [#36756](https://github.com/sgl-project/sglang/pull/36756) Limit concurrent build jobs for cu134 container
- [#36568](https://github.com/sgl-project/sglang/pull/36568) perf: skip redundant scheduler metadata gather for DP1
- [#36637](https://github.com/sgl-project/sglang/pull/36637) [mem_cache] Add `free_full` to release the full side of a tombstoned SWA node
- [#36211](https://github.com/sgl-project/sglang/pull/36211) [k3] declare packed_modules_mapping on `KimiK3ForConditionalGeneration`
- [#36760](https://github.com/sgl-project/sglang/pull/36760) [sglang-miles] Cherry pick #35708
- [#35374](https://github.com/sgl-project/sglang/pull/35374) [Kernel] Add H200 MoE configs for Qwen3.5 and Qwen3.6

---
*Auto-generated by npu_pr_monitor.py*