# NPU PR 监控报告 (已合入)
**生成时间**: 2026-08-25 23:46 UTC
**本次检查已合入 PR 数**: 83
**涉及 NPU**: 16 | **无关**: 67 | **不确定**: 0

---

## ⚠️ 涉及 NPU 的已合入 PR

### [#36142](https://github.com/sgl-project/sglang/pull/36142) [AMD][CI] Add MiniMax-M3-MXFP8 MI35x nightly perf benchmark
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#25540](https://github.com/sgl-project/sglang/pull/25540) Use DeepGEMM BF16 for unquantized DeepEP LL MoE
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#29619](https://github.com/sgl-project/sglang/pull/29619) [DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#29775](https://github.com/sgl-project/sglang/pull/29775) [DeepSeek V4] Enable FlashMLA sparse prefill by default
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 5

### [#12964](https://github.com/sgl-project/sglang/pull/12964) [DeepseekV3.2] Deepseek fp8 support for MHA path
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 2

### [#18917](https://github.com/sgl-project/sglang/pull/18917) [Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#10788](https://github.com/sgl-project/sglang/pull/10788) Fix: Dynamic RoPE Cache Expansion to Prevent Position-ID Out-of-Bounds in EAGLE + Long-Sequence Workloads
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#29069](https://github.com/sgl-project/sglang/pull/29069) fix(runner): autotune flashinfer MoE on a decode-shaped buffer
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

### [#2700](https://github.com/sgl-project/sglang/pull/2700) Feature/function calling update
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 10

### [#29734](https://github.com/sgl-project/sglang/pull/29734) [GDN] Auto-select FlashInfer GDN prefill on validated SM100 configs
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 6

### [#25002](https://github.com/sgl-project/sglang/pull/25002) [spec_v2] Enable trtllm_mha draft-extend CUDA graph with v2 semantics
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#29988](https://github.com/sgl-project/sglang/pull/29988) [dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 3

### [#25195](https://github.com/sgl-project/sglang/pull/25195) [BCG] Support breakable CUDA graph for DeepSeek V4 DP attention
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 13

### [#13646](https://github.com/sgl-project/sglang/pull/13646) [DeepSeekV3.2] Enable pure TP & Partial DP Attention
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 7

### [#30620](https://github.com/sgl-project/sglang/pull/30620) Allow prefill breakable CUDA graph for Qwen3.5 via multimodal opt-in allowlist
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 4

### [#36257](https://github.com/sgl-project/sglang/pull/36257) [NPU] [CI] Split the base-c-test-acc-2-npu-a3 task into two parts running in parallel
- **检测方式**: 关键词匹配(标题+文件双命中)
- **理由**: 标题和文件均命中 NPU 关键词
- **文件数**: 1

## ✅ 与 NPU 无关的已合入 PR
- [#36381](https://github.com/sgl-project/sglang/pull/36381) Fix SWA ownership across grouped frees
- [#35505](https://github.com/sgl-project/sglang/pull/35505) [Deepseek-V4] Enable shared-experts fusion on the flashinfer_mxfp4 (trtllm-gen) MoE path
- [#36351](https://github.com/sgl-project/sglang/pull/36351) fix(disagg): snapshot affected rooms before iterating outside the lock
- [#36029](https://github.com/sgl-project/sglang/pull/36029) fix(disagg): refresh stale prefill bootstrap metadata
- [#36342](https://github.com/sgl-project/sglang/pull/36342) docs(cookbook): use auto parser resolution for Granite 4.2
- [#36290](https://github.com/sgl-project/sglang/pull/36290) [AMD][CI] Adjust MI300 score API performance thresholds
- [#24378](https://github.com/sgl-project/sglang/pull/24378) fix(disagg): broadcast bootstrap port across multi-node prefill ranks
- [#30140](https://github.com/sgl-project/sglang/pull/30140) [DeepSeek-V4] Enable non-paged indexer by default for large prefill chunks
- [#36219](https://github.com/sgl-project/sglang/pull/36219) [Performance] Tune FlashInfer EXTEND for DP prefill
- [#16463](https://github.com/sgl-project/sglang/pull/16463) [test] update acc len threshold to 2.7 for eagle dp attention tests
- [#19890](https://github.com/sgl-project/sglang/pull/19890) [Disagg] GPU staging buffer with dynamic ring allocator for heterogeneous TP KV transfer
- [#16974](https://github.com/sgl-project/sglang/pull/16974) [SPEC_V2] Enable cudagraph draft_extend for trtllm_mla_backend and Acclen Fix for DP under cudagraph mode
- [#17041](https://github.com/sgl-project/sglang/pull/17041) [eval] GSM8k support for run_eval
- [#21861](https://github.com/sgl-project/sglang/pull/21861)   [GDN] Remove FlashInfer GDN decode + no_buffer guard and default to FlashInfer on SM100+  
- [#19169](https://github.com/sgl-project/sglang/pull/19169) [Qwen3.5] Raise Exception when radix_cache and extra_buffer are enabled at the same time
- [#19002](https://github.com/sgl-project/sglang/pull/19002) [Fix][Qwen3.5] Pass max_mamba_cache_size to mamba pool in disaggregation decode path
- [#19350](https://github.com/sgl-project/sglang/pull/19350) [Logging] Fix prefill side logging in pd disagg
- [#19076](https://github.com/sgl-project/sglang/pull/19076) [Fix] Quick fix for int32 overflow in Mooncakes' send_kvcache_slice
- [#25859](https://github.com/sgl-project/sglang/pull/25859) [DSA] Make MQA logits free memory ratio configurable
- [#25299](https://github.com/sgl-project/sglang/pull/25299) [NSA] Avoid repeated NSA MQA logits memory queries
- [#15086](https://github.com/sgl-project/sglang/pull/15086) [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache
- [#15790](https://github.com/sgl-project/sglang/pull/15790) [MTP][spec_v2] Fix TRTLLM MLA backend crash in EAGLE draft_extend mode 
- [#15027](https://github.com/sgl-project/sglang/pull/15027) [PP Prefill][NIXL] Fix PP mode transfer completion tracking to wait for all ranks
- [#26238](https://github.com/sgl-project/sglang/pull/26238) refactor(dsv4): route MHC prenorm through DeepGEMM wrapper
- [#12788](https://github.com/sgl-project/sglang/pull/12788) [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100)
- [#20655](https://github.com/sgl-project/sglang/pull/20655) [Qwen3.5] mamba slice fix (Prefill TP != Decode TP & decode TP size>1)
- [#11126](https://github.com/sgl-project/sglang/pull/11126) Optimize copy_kv_cache for spec decoding
- [#13718](https://github.com/sgl-project/sglang/pull/13718) Upgrade flashmla kernel for NSA tp support
- [#15227](https://github.com/sgl-project/sglang/pull/15227) [IDLE FORWARD][Indexer] Fix forward_idle bs mismatch issue in DeepseekV3.2's NSAIndexer
- [#30997](https://github.com/sgl-project/sglang/pull/30997) [Disagg][Qwen3.5] Fix heterogeneous attn-TP scatter transfer: GDN conv sub-block slice + GQA replicated-KV head map
- [#12115](https://github.com/sgl-project/sglang/pull/12115) Fix Illegal Instruction/IMA errors when using DP attention -- num_tokens_for_logprob calculation
- [#11871](https://github.com/sgl-project/sglang/pull/11871) Fix: Safe RoPE Cache Expansion to Prevent Position-ID Out-of-Bounds in EAGLE + Long-Sequence Workloads
- [#27747](https://github.com/sgl-project/sglang/pull/27747) fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay
- [#24906](https://github.com/sgl-project/sglang/pull/24906) Support Qwen3.5 NVFP4 MTP DeepEP
- [#27868](https://github.com/sgl-project/sglang/pull/27868) fix(qwen3.5): keep CUDA dual-stream overlap (regressed by #25885)
- [#14245](https://github.com/sgl-project/sglang/pull/14245) Fix NSA Bug in Centralize NSA Dispatch Logic
- [#23773](https://github.com/sgl-project/sglang/pull/23773) [fix] nixl: transport SWA/NSA/Mamba state buffer
- [#22145](https://github.com/sgl-project/sglang/pull/22145) [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support)
- [#27945](https://github.com/sgl-project/sglang/pull/27945) fix(moe): make FlashInfer A2A robust to collapsed global_num_tokens (moe_dense_tp_size NaN)
- [#18500](https://github.com/sgl-project/sglang/pull/18500) [Flashinfer Autotune] Fix FlashInfer FP4 MoE autotuning crash by removing incorrect flatten on hidden_states_scale
- [#24856](https://github.com/sgl-project/sglang/pull/24856) Fix TRTLLM MHA routing for draft extend
- [#24785](https://github.com/sgl-project/sglang/pull/24785) Fix reduce_scatterv producer contract for SUM_LEN
- [#14325](https://github.com/sgl-project/sglang/pull/14325) [DeepseekV3.2][NSA][Indexer] Fix PAGED top-k transform for NSA indexer chunked execution on H200
- [#21921](https://github.com/sgl-project/sglang/pull/21921) Add staging buffer CI test and documentation for heterogeneous TP
- [#23189](https://github.com/sgl-project/sglang/pull/23189) feat(scheduler): add adaptive queue-based prefill delayer trigger
- [#22536](https://github.com/sgl-project/sglang/pull/22536) [Disagg][NIXL] Add staging buffer support for heterogeneous TP KV transfer
- [#22240](https://github.com/sgl-project/sglang/pull/22240) [Disagg][NIXL] Support Mamba state slice transfer for heterogeneous TP (Step 2/2 for Qwen3.5)
- [#22642](https://github.com/sgl-project/sglang/pull/22642) Replace all-reduce + dp_scatter with reduce_scatterv for DP attention
- [#27954](https://github.com/sgl-project/sglang/pull/27954) [dsv4] Pad MLA decode q-heads to 64 (not full n_heads) for FlashMLA head64 kernel
- [#25810](https://github.com/sgl-project/sglang/pull/25810) perf(dsv4): add MHC token-count prewarm
- [#27986](https://github.com/sgl-project/sglang/pull/27986) [dsv4] Prewarm MHC prenorm kernel at startup
- [#11892](https://github.com/sgl-project/sglang/pull/11892) DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill
- [#16310](https://github.com/sgl-project/sglang/pull/16310) [SPEC_V2] Fix Acclen drop when enabling DP Attention for Spec-Overlap
- [#12868](https://github.com/sgl-project/sglang/pull/12868) [Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill
- [#19086](https://github.com/sgl-project/sglang/pull/19086) [Fix][Qwen3.5] Fix KV cache slice transfer for GQA models with replicated KV heads
- [#30443](https://github.com/sgl-project/sglang/pull/30443) [NVIDIA] Allow modelopt_mixed quantization with flashinfer_cutedsl MoE runner
- [#15111](https://github.com/sgl-project/sglang/pull/15111) [EAGLE] Fix slow Triton compilation in EAGLE KV cache copy by chunking large num_locs_upper
- [#13544](https://github.com/sgl-project/sglang/pull/13544) [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend
- [#17051](https://github.com/sgl-project/sglang/pull/17051) [ConfigArgumentMerger] Improve ConfigArgumentMerger compatibility with external callers
- [#36286](https://github.com/sgl-project/sglang/pull/36286) docs(cookbook): add IBM Granite 4.2 cookbook
- [#36097](https://github.com/sgl-project/sglang/pull/36097) Fix MXFP8 MoE weight sizing for non-gated models
- [#35188](https://github.com/sgl-project/sglang/pull/35188) [Bugfix] Fix int32 destination offset overflow in CUTLASS MoE pre-reorder
- [#36169](https://github.com/sgl-project/sglang/pull/36169) [diffusion] docs: desktop-safe 24 GB recipe and the DGX Spark tier
- [#36246](https://github.com/sgl-project/sglang/pull/36246) [AMD] Add Kimi-K2.7-Code-MXFP4 to cookbook
- [#36300](https://github.com/sgl-project/sglang/pull/36300) config: the model-config cache keys on the path the record carried
- [#33021](https://github.com/sgl-project/sglang/pull/33021) [AMD] Drop redundant FP8 bpreshuffle scale transpose via fused AR kernel
- [#35672](https://github.com/sgl-project/sglang/pull/35672) [AMD] Enable draft_extend CUDA graph for HIP DSA backend

---
*Auto-generated by npu_pr_monitor.py*