# NPU PR 监控报告 (已合入)
**生成时间**: 2026-09-24 23:00 UTC
**本次检查已合入 PR 数**: 47
**涉及 NPU**: 17 | **无关**: 4 | **不确定**: 26

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#37442](https://github.com/sgl-project/sglang/pull/37442) Add 8-node AllReduce/AllGather and MNVLS algorithm support to MSCCL++
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 8

### [#36559](https://github.com/sgl-project/sglang/pull/36559) MoE: small-batch sorting path with fused mxfp8 quantisation
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#36574](https://github.com/sgl-project/sglang/pull/36574) MiniMax-M3: MXFP8 dense-only block convert + aiter MXFP8 MoE on gfx950
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#41084](https://github.com/sgl-project/sglang/pull/41084) [Refactor] Split prepare_attn into a reduction step and per-quant-format residual steps
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#41097](https://github.com/sgl-project/sglang/pull/41097) [Refactor] Share the MoE output all-reduce between models
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 21

### [#41081](https://github.com/sgl-project/sglang/pull/41081) [Refactor] Pass each layer stack's output through a communicator exit
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 33

### [#41080](https://github.com/sgl-project/sglang/pull/41080) [Fix] Complete the deferred FFN all-reduce before deepstack addition and aux hidden-state capture
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 11

### [#41079](https://github.com/sgl-project/sglang/pull/41079) [Fix] Complete the deferred FFN all-reduce before a pipeline-parallel send
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 23

### [#39478](https://github.com/sgl-project/sglang/pull/39478) Support unified memory decode host pools
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 24

### [#41023](https://github.com/sgl-project/sglang/pull/41023) [PD] Enable deferred decode-side KV release by default
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#40922](https://github.com/sgl-project/sglang/pull/40922) [Refactor] Retire the model-specific Kimi K3 kernel namespace
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 52

### [#40524](https://github.com/sgl-project/sglang/pull/40524) [NPU] Update CANN version to 9.1.0
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 15

### [#40388](https://github.com/sgl-project/sglang/pull/40388) [Diffusion] Enable lossless SANA-Video eager conv fusions for 12.6% lower latency
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#40425](https://github.com/sgl-project/sglang/pull/40425) [Diffusion] Fuse lossless LingBot World FP32 normalization
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#40405](https://github.com/sgl-project/sglang/pull/40405) [Diffusion] Fuse lossless Wan VAE post-ops for LongLive 2 I2V
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#40384](https://github.com/sgl-project/sglang/pull/40384) [Diffusion] Fuse LongCat GELU+cat and support Edit-Turbo BCG
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 10

### [#40826](https://github.com/sgl-project/sglang/pull/40826) [Feature] Add per-item candidate token scoring and calibration
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 9

## ❓ 不确定是否涉及 NPU 的 PR

### [#41179](https://github.com/sgl-project/sglang/pull/41179) Fix mixed chunk prefill with DP speculative coordination
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39929](https://github.com/sgl-project/sglang/pull/39929) [Bugfix] Align DeepSeek-V4.1 reasoning effort budgets
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41049](https://github.com/sgl-project/sglang/pull/41049) [DSV4] Size compressed pools from one per-ratio table in DSV4PoolConfigurator
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41046](https://github.com/sgl-project/sglang/pull/41046) [Docs] Enable Qwen3.8 Flash Next NVIDIA NVFP4 on B200/B300/GB300
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41090](https://github.com/sgl-project/sglang/pull/41090) [DSV4] Fix TRTLLM uniform FP8 KV memory budgeting
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40041](https://github.com/sgl-project/sglang/pull/40041) [qwen 3.8 next] Fuse Qwen PLE gate and convolution preparation for target verify
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41120](https://github.com/sgl-project/sglang/pull/41120) [AMD] Add .co for deepseek v4 fp8 decode kernel and add group decode opt
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41083](https://github.com/sgl-project/sglang/pull/41083) [Fix] Broadcast requests along attention CP before attention TP
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41082](https://github.com/sgl-project/sglang/pull/41082) [Fix] Step-3.5: stop dense layers from summing their output twice under DP attention
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40858](https://github.com/sgl-project/sglang/pull/40858) [DP attention] Publish DP buffer sizes from a ForwardBatch
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41138](https://github.com/sgl-project/sglang/pull/41138) [Fix] Skip the DCP target-verify MLA kernel during FlashInfer autotune
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#2601](https://github.com/sgl-project/sglang/pull/2601) [Feature, Hardware] Enable DeepseekV3 on AMD GPUs
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40805](https://github.com/sgl-project/sglang/pull/40805) fix: Triton 3.8 compatbility to support DSV4.1-Flash in CUDA 13.4 image (Rubin)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#38726](https://github.com/sgl-project/sglang/pull/38726) [Quant] ModelOpt mixed precision: dispatch block-FP8 MoE experts and derive the block size
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40712](https://github.com/sgl-project/sglang/pull/40712) [HiCache] Demote internal-node SWA KV to host on write_back eviction instead of dropping it
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#28960](https://github.com/sgl-project/sglang/pull/28960) fix(sampling): validate sampling_seed is an int within int64 range
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40866](https://github.com/sgl-project/sglang/pull/40866) [chore] surface the cookbook to users who pip install sglang
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41130](https://github.com/sgl-project/sglang/pull/41130) [Diffusion] Add @niehen6174 as a code owner
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40512](https://github.com/sgl-project/sglang/pull/40512) [HiCache] Make host reclamation independent of transfer order
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39816](https://github.com/sgl-project/sglang/pull/39816) Refactor the Cute-DSL AR fusion to support DeepseekV2 archs (GLM-5.3, etc.)
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40612](https://github.com/sgl-project/sglang/pull/40612) [Diffusion] migrate the whole _register_configs from registry.py to the model own config file
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40996](https://github.com/sgl-project/sglang/pull/40996) [AMD] Add tuned dsv4 shape
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#40878](https://github.com/sgl-project/sglang/pull/40878) [AMD][DSV4] fp8 unified_kv decode: wave-aware split count past 40 tokens
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#39059](https://github.com/sgl-project/sglang/pull/39059) [AMD] Tune Triton sparse MLA on gfx950 and make split-K workspaces graph-safe
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41096](https://github.com/sgl-project/sglang/pull/41096) Fix TBO child batch missing dp_spec_prefill_coordination_applied
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

### [#41078](https://github.com/sgl-project/sglang/pull/41078) Read a per-replica sequence at the slot of the gather that produced it
- **理由**: AI 调用失败: 402 Client Error: Payment Required for url: https://api.deepseek.com/v1/chat/completions

## ✅ 与 NPU 无关的已合入 PR
- [#41162](https://github.com/sgl-project/sglang/pull/41162) [Fix] Patch set_dp_buffer_len_from_batch in DP spec prefill coordination test
- [#41109](https://github.com/sgl-project/sglang/pull/41109) [AMD] GLM-5.2 MI355X MXFP4: bump image to 20260923 daily
- [#41155](https://github.com/sgl-project/sglang/pull/41155) [Test] Run the Qwen3.5 Triton DCP nightly with the radix cache enabled
- [#40387](https://github.com/sgl-project/sglang/pull/40387) [AMD] ci: move the Miles ROCm 7.2 nightly build to 7.2.4

---
*Auto-generated by npu_pr_monitor.py*