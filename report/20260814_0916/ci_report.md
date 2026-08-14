# NPU CI 执行监控
**生成时间**: 2026-08-14 01:16 UTC
**分析 Run 数**: 75

---

## 📊 本次执行总结

- **成功 Job 数**: 435
- **失败 Run 数**: 73
- **成功 Job 平均耗时**: 27.3min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 272.4min | #31715508299 | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94546447068) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 137.6min | #31718301030 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631789440) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 126.5min | #31724431325 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529302147) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 126.2min | #31709151289 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422933) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.3min | #31721168304 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522890123) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.7min | #31714664459 | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864800) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.6min | #31718403338 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188562) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.1min | #31718370243 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508896561) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.8min | #31707526148 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471842500) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.4min | #31709621803 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478979386) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 结论 | 链接 |
|--------|------|------|-----------|------|------|
| #31715508299<br>[#34542 [MiniMax-M3] Overlap shared and routed experts](https://github.com/sgl-project/sglang/pull/34542) | `minimax-m3-moe-dual-stream` | 463.8min | 1 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31715508299) |
| #31718370243<br>[#30371 [DSV4] Fix SWA state pool over-allocation by using storage page size instead of model window](https://github.com/sgl-project/sglang/pull/30371) | `dsv4_state_pool_size` | 311.2min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31718370243) |
| #31718123655<br>[#34729 Retain SWA down to the last state checkpoint](https://github.com/sgl-project/sglang/pull/34729) | `swa-retain-to-mamba-checkpoint` | 295.1min | 3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31718123655) |
| #31718301030<br>[#30318 [NPU] Add mxfp4-w4a8 MOE Quantization Support for NPU](https://github.com/sgl-project/sglang/pull/30318) | `add_mxfp4w4a8_quantization_for_npu` | 267.9min | 3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31718301030) |
| #31718403338<br>[#34749 feat(rust-server): add model extension hooks](https://github.com/sgl-project/sglang/pull/34749) | `lmzheng/model-extension-hooks` | 266.0min | 3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31718403338) |
| #31714664459<br>[#34736 [Diffusion] Unify component residency controls](https://github.com/sgl-project/sglang/pull/34736) | `codex/component-residency-policy` | 256.6min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31714664459) |
| #31724431325<br>[#34355 [XPU] Support decode context parallelism (DCP) on Intel XPU](https://github.com/sgl-project/sglang/pull/34355) | `xpu-dcp-support` | 255.4min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31724431325) |
| #29403277122<br>[#29326 feat(hicache): Add shared memory allocator for host KV cache](https://github.com/sgl-project/sglang/pull/29326) | `hicache-shm-allocator` | 255.2min | 3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29403277122) |
| #29289186291<br>[#28416 [GLM5][MoE] perf: Write FlashInfer TRT-LLM MoE output directly](https://github.com/sgl-project/sglang/pull/28416) | `glm5/moe-output-output` | 249.3min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29289186291) |
| #31709151289<br>[#29723 [AMD] Add fused all-reduce RMSNorm per-token FP8/MXFP4 quant](https://github.com/sgl-project/sglang/pull/29723) | `marv/ar_norm_per_token_quant_fusion` | 246.7min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31709151289) |
| #29404997476<br>[#31298 fix: warm up Kimi VLM vision encoder at startup](https://github.com/sgl-project/sglang/pull/31298) | `codex/kimi-vlm-warmup` | 245.2min | 3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29404997476) |
| #29392287525 | `feat/sm120_glm51` | 245.1min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29392287525) |
| #31721168304<br>[#34753 feat(cli): add extensible serve backend plugins](https://github.com/sgl-project/sglang/pull/34753) | `codex/extensible-serve-backends` | 240.7min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31721168304) |
| #29402588321<br>[#31307 [Kernel] Fill non-CUDA coverage: HIP (aiter/rocm-triton) + Ascend NPU backends (RFC #29630)](https://github.com/sgl-project/sglang/pull/31307) | `bbuf/kernels-fill-noncuda-coverage` | 240.2min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29402588321) |
| #31718371605<br>[#33857 [Perf] Skip trivial DSV4 nonpaged indexer logits](https://github.com/sgl-project/sglang/pull/33857) | `perf/dsv4-nonpaged-trivial-rows` | 236.9min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31718371605) |
| #31707526148<br>[#30805 [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103](https://github.com/sgl-project/sglang/pull/30805) | `dsv4_fp8_trtllm_gen` | 220.0min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31707526148) |
| #31711897573<br>[#33068 [AMD] Fuse quantized in_proj layers in Qwen3.5](https://github.com/sgl-project/sglang/pull/33068) | `marv/fuse_gdn_in_proj` | 215.3min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31711897573) |
| #31722157936<br>[#29668 [HiCache] fix: resolve Mooncake local_hostname per node for runtime attach](https://github.com/sgl-project/sglang/pull/29668) | `cursor/fix-mooncake-local-hostname-20a4` | 212.2min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31722157936) |
| #31706940714<br>[#24911 Profiling Enhancements [2/3]: detailed execution step annotations](https://github.com/sgl-project/sglang/pull/24911) | `feat/roofline_annotations` | 211.8min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31706940714) |
| #31709621803<br>[#34691 fix: add missing backend key to Kimi-K3 deferred GPU preprocessing config](https://github.com/sgl-project/sglang/pull/34691) | `mmangkad/fix-kimi-k3-deferred-backend-key` | 205.9min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31709621803) |
| #31727270352<br>[#27770 [P/D disagg] Decode-side radix cache for SWA hybrid models (unified radix tree)](https://github.com/sgl-project/sglang/pull/27770) | `idhanani/unified-radix-swa-fix` | 198.7min | 4 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31727270352) |
| #31725107558<br>[#32514 feat(kv-events): Add component_types field to BlockStored for per-component placement tracking](https://github.com/sgl-project/sglang/pull/32514) | `feat/kv-events-component-placement-v2` | 193.3min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31725107558) |
| #31727027724<br>[#33604 Fix Whisper transcription for audio over 30 seconds](https://github.com/sgl-project/sglang/pull/33604) | `agent/whisper-long-audio-chunking` | 189.3min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31727027724) |
| #29339195925<br>[#23317 [Bug Fix] Sync FlashInfer autotune tactic selection across TP ranks](https://github.com/sgl-project/sglang/pull/23317) | `htphan/fix-symm-mem-cuda-graph-deadlock` | 182.6min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29339195925) |
| #29340140589<br>[#28836 [Deps] Upgrade CUDA PyTorch stack to 2.13](https://github.com/sgl-project/sglang/pull/28836) | `mmangkad/torch-2.12` | 178.2min | 3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29340140589) |
| #29339745832<br>[#31057 feat(mem_cache): semantic KV cache reuse via a pluggable fuzzy-match radix backend](https://github.com/sgl-project/sglang/pull/31057) | `feat/semantic-radix-backend` | 177.7min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29339745832) |
| #29337703742<br>[#30792 [Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7)](https://github.com/sgl-project/sglang/pull/30792) | `kernels/phase25-dsa-dsv4` | 176.4min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29337703742) |
| #29318774860<br>[#30947 [1/3] [EAGLE] perf: Fuse topk=1 draft postprocess](https://github.com/sgl-project/sglang/pull/30947) | `glm52/mtp-split-1-topk1` | 174.7min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29318774860) |
| #29338486836 | `pr_add_multi_stream_gemm_fusion` | 172.2min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29338486836) |
| #29338080660<br>[#30795 [Kernel] Relocate vendored fla and mamba kernel trees to sglang.kernels (RFC #29630, Phase 2.5, 7/7)](https://github.com/sgl-project/sglang/pull/30795) | `kernels/phase25-vendored` | 172.2min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29338080660) |
| #29329839778<br>[#28370 Fix invalid escape warnings in tool parsers](https://github.com/sgl-project/sglang/pull/28370) | `fix/glm-tool-parser-escapes` | 171.7min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29329839778) |
| #29328209895<br>[#31131 [AMD] Fix DSV4 JIT build on rocm ](https://github.com/sgl-project/sglang/pull/31131) | `amd_fix_deepseekv4_0714` | 167.5min | 1 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29328209895) |
| #29330274051<br>[#31029 [Diffusion] post_training: Add LoRA IPC weight sync via lora_merge mode](https://github.com/sgl-project/sglang/pull/31029) | `feat/lora-merge-ipc-update` | 162.4min | 1 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29330274051) |
| #31713837840<br>[#34517 [AMD][Spec] Accelerate Qwen3.5 verification with grouped-head shared KV](https://github.com/sgl-project/sglang/pull/34517) | `feat/qwen35-shared-kv-verify` | 158.1min | 4 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31713837840) |
| #29337380180 | `fuse-gate-gemv-into-append` | 153.4min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29337380180) |
| #29319427123 | `pr_add_multi_stream_gemm_fusion` | 152.4min | 3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29319427123) |
| #29319625597<br>[#31173 [PD] Stride KV token->page indices on device before D2H copy](https://github.com/sgl-project/sglang/pull/31173) | `cctry/kv-to-page-indices-on-device` | 151.8min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29319625597) |
| #29321466158<br>[#30793 [Kernel] Migrate linear-attention, MiniMax-sparse and diffusion kernels to sglang.kernels (RFC #29630, Phase 2.5, 6/7)](https://github.com/sgl-project/sglang/pull/30793) | `kernels/phase25-linear` | 141.1min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29321466158) |
| #29321089757<br>[#31177 [Diffusion] Support fal Ideogram V4 Fast and Instant](https://github.com/sgl-project/sglang/pull/31177) | `codex/support-fal-ideogram-v4-fast-instant` | 140.3min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29321089757) |
| #29322097405<br>[#31042 [CI] Fix SGLANG_JIT_KERNEL_RUN_FULL_TESTS never activating the nightly full jit-kernel sweep](https://github.com/sgl-project/sglang/pull/31042) | `main` | 139.5min | 7 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29322097405) |
| #29397784488<br>[#30682 [BugFix] Preserve tokenizer worker fanout when `skip_tokenizer_init` is enabled](https://github.com/sgl-project/sglang/pull/30682) | `skip_tokenizer_multi_worker` | 108.7min | 2 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29397784488) |
| #31727209003<br>[#34755 [CI][PD] Pin nccl rendezvous port per side to fix flaky disaggregation tests](https://github.com/sgl-project/sglang/pull/34755) | `main` | 102.2min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31727209003) |
| #29411683974<br>[#18263 [AMD][AITER attention backend] Adjust `available_bytes` in `KVCacheConfigurator` to avoid OOMs in AITER attention backend buffers allocation](https://github.com/sgl-project/sglang/pull/18263) | `fix-kv-cache-aiter-memory-allocation` | 84.6min | 5 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29411683974) |
| #29345312974<br>[#29690 Fuse the preprocess kernels of trtllm-gen attention](https://github.com/sgl-project/sglang/pull/29690) | `brayden/fuse-trtllm-gen-prologue-kernels` | 77.6min | 5 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29345312974) |
| #29344715585<br>[#30280 Delete sgl-kernel AOT router GEMM and fused A GEMM](https://github.com/sgl-project/sglang/pull/30280) | `brayden/remove-aot-router-fused-a-gemm` | 77.4min | 5 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29344715585) |
| #31707437509<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 71.0min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31707437509) |
| #29345341583<br>[#31109 Remove QServe and FBGEMM FP8 quantization](https://github.com/sgl-project/sglang/pull/31109) | `remove-qserve-quantization` | 69.5min | 5 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29345341583) |
| #29202367445<br>[#30833 Allocate single-node DP-attention ports from a free block to avoid collisions](https://github.com/sgl-project/sglang/pull/30833) | `dp-attn-free-port-block` | 60.7min | 2 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29202367445) |
| #31713616678<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 59.8min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31713616678) |
| #29193506426 | `tom_refactor_202605a/primary/nonmech_model_runner` | 56.7min | 8 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29193506426) |
| #31710585087<br>[#34620 [Diffusion][ERNIE] Fuse QKNorm with full-width RoPE](https://github.com/sgl-project/sglang/pull/34620) | `bbuf/b300-ernie-qknorm-rope` | 53.0min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31710585087) |
| #31713481119<br>[#34729 Retain SWA down to the last state checkpoint](https://github.com/sgl-project/sglang/pull/34729) | `swa-retain-to-mamba-checkpoint` | 52.1min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31713481119) |
| #31721938655<br>[#32405 [MoE Refactor] Migrate SM100 trtllm-gen mxfp4 MoE onto MoeRunner](https://github.com/sgl-project/sglang/pull/32405) | `refactor-mxfp4-sm100-trtllm-moerunner` | 47.0min | 12 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31721938655) |
| #29197701797 | `tom_refactor_202605a/primary/nonmech_model_runner` | 44.0min | 9 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29197701797) |
| #29201063345 | `jit_dsv4_c128_opt` | 42.4min | 1 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/29201063345) |
| #31719178903<br>[#32926 [AMD] Don't request the unused softmax LSE in the AITER diffusion backend](https://github.com/sgl-project/sglang/pull/32926) | `fix/aiter-diffusion-drop-unused-lse` | 42.4min | 1 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31719178903) |
| #29339049315 | `idhanani/unified-radix-swa-fix` | 37.0min | 6 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29339049315) |
| #31724189348<br>[#34692 [PD] Add the missing Prefill bootstrap timeout for NIXL](https://github.com/sgl-project/sglang/pull/34692) | `main` | 36.6min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31724189348) |
| #31724230737<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 36.1min | 10 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31724230737) |
| #29338838523 | `bbuf/hpc-ops-attention-backend` | 36.0min | 6 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29338838523) |
| #29350702274 | `jialino/radix-cache-split` | 32.9min | 6 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29350702274) |
| #31712200105<br>[#34736 [Diffusion] Unify component residency controls](https://github.com/sgl-project/sglang/pull/34736) | `codex/component-residency-policy` | 32.4min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31712200105) |
| #31721830435<br>[#33827 fix: make Cache-DiT actually cache on MiniMax-H3](https://github.com/sgl-project/sglang/pull/33827) | `main` | 28.6min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31721830435) |
| #29201454263<br>[#30838 [JIT] Refactor dtype traits into DTypeTrait and unify warp reductions](https://github.com/sgl-project/sglang/pull/30838) | `jit-dtype-trait-reduce-fix` | 26.1min | 1 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29201454263) |
| #31722141924<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 25.4min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31722141924) |
| #31719494392<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 21.9min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31719494392) |
| #31727235228<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 21.2min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31727235228) |
| #31726053280<br>[#27770 [P/D disagg] Decode-side radix cache for SWA hybrid models (unified radix tree)](https://github.com/sgl-project/sglang/pull/27770) | `idhanani/unified-radix-swa-fix` | 15.1min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31726053280) |
| #29199118191 | `tom_refactor_202605a/primary/nonmech_model_runner` | 12.8min | 9 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29199118191) |
| #29193175682<br>[#30927 Gate Rust extension builds](https://github.com/sgl-project/sglang/pull/30927) | `main` | 10.2min | 9 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29193175682) |
| #31721323279<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 9.9min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31721323279) |
| #29342296396 | `bbuf/hpc-ops-attention-backend` | 9.5min | 9 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/29342296396) |
| #31718945455<br>[#34608 Publish per-scheduler load on a dedicated socket for load-aware routers](https://github.com/sgl-project/sglang/pull/34608) | `sgl-router/upstream-lb-1-load-publisher` | 7.1min | 11 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31718945455) |

---


## [Run #29411683974](https://github.com/sgl-project/sglang/actions/runs/29411683974)
- **分支**: `fix-kv-cache-aiter-memory-allocation`
- **总耗时**: 84.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29411683974

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073760) |
| stage-b-test-16-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073807) |
| multimodal-gen-test-1-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073818) |
| multimodal-gen-test-2-npu-a3 | 84.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073835) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 84.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340074158) |

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073760

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073807

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073818

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073835

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明CI作业尝试访问的远程存储对象缺失，可能是构建产物或依赖文件未正确上传，或路径配置错误，属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340074158

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073794) |
| stage-b-test-2-npu-a2 (0) | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073812) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073824) |
| stage-b-test-2-npu-a2 (1) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29411683974/job/87340073893) |


## [Run #29404997476](https://github.com/sgl-project/sglang/actions/runs/29404997476)
- **分支**: `codex/kimi-vlm-warmup`
- **总耗时**: 245.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29404997476

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254336) |
| multimodal-gen-test-2-npu-a3 | 62.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254342) |
| stage-b-test-4-npu-a3 | 48.6min | 其他 | 测试用例 test_npu_llada2_mini.py 失败，其余4个用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254461) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、checkout、上传artifact等步骤，未显示multimodal-gen测试的实际执行和失败原因，可能因日志截断或作业在测试前被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254336

- **multimodal-gen-test-2-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254342

- **stage-b-test-4-npu-a3**: 作业中5个NPU测试有4个通过，仅 test_npu_llada2_mini.py 返回退出码1，耗时895秒。日志未显示具体错误原因，可能为用例本身问题或环境相关，需进一步查看该用例详细输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254461

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254296) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254297) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254345) |
| stage-b-test-1-npu-a2 (1) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254357) |
| stage-b-test-16-npu-a3 | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254376) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29404997476/job/87318254851) |


## [Run #29403277122](https://github.com/sgl-project/sglang/actions/runs/29403277122)
- **分支**: `hicache-shm-allocator`
- **总耗时**: 255.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29403277122

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 46.7min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，4/5通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759004) |
| multimodal-gen-test-2-npu-a3 | 63.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759028) |
| multimodal-gen-test-1-npu-a3 | 62.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759044) |

- **stage-b-test-4-npu-a3**: 测试test_npu_llada2_mini.py返回退出码1，耗时855秒，其余4个测试均通过。该测试属于dllm功能，可能涉及LLaDA2模型推理逻辑或环境配置问题，需查看具体错误日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759004

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node 20弃用警告、上传diffusion-failures目录时提示无文件，以及清理步骤。可能因日志截断或作业在早期阶段被取消，需查看完整日志定位真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759028

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但实际失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759044

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759012) |
| stage-b-test-16-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759040) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759052) |
| stage-b-test-1-npu-a2 (0) | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759062) |
| stage-b-test-1-npu-a2 (1) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759064) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29403277122/job/87312759549) |


## [Run #29402588321](https://github.com/sgl-project/sglang/actions/runs/29402588321)
- **分支**: `bbuf/kernels-fill-noncuda-coverage`
- **总耗时**: 240.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29402588321

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 48.3min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，4/5通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494369) |
| multimodal-gen-test-1-npu-a3 | 54.0min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含GitHub Actions基础设施信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494407) |

- **stage-b-test-4-npu-a3**: 测试test_npu_llada2_mini.py返回退出码1，耗时925秒，其余4个测试均通过。该测试属于dllm功能模块，可能因代码逻辑错误或环境配置问题导致失败，需查看具体错误日志定位。
  链接: https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494369

- **multimodal-gen-test-1-npu-a3**: 日志仅包含runner启动、checkout、upload-artifact等基础设施步骤，未显示实际测试执行和失败信息。上传diffusion-failures目录时提示无文件，说明测试可能未运行或失败原因未记录。
  链接: https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494407

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494370) |
| stage-b-test-1-npu-a2 (0) | 43.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494402) |
| multimodal-gen-test-2-npu-a3 | 39.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494419) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494426) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494536) |
| stage-b-test-16-npu-a3 | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494580) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29402588321/job/87310494818) |


## [Run #29397784488](https://github.com/sgl-project/sglang/actions/runs/29397784488)
- **分支**: `skip_tokenizer_multi_worker`
- **总耗时**: 108.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29397784488

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 108.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337067359) |
| stage-b-test-4-npu-a3 | 108.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337067381) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中缺失，可能是文件被误删、路径错误或上传未完成，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337067359

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337067381

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068068) |
| stage-b-test-16-npu-a3 | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068193) |
| stage-b-test-1-npu-a2 (1) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068214) |
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068316) |
| multimodal-gen-test-2-npu-a3 | 35.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068352) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068415) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29397784488/job/87337068466) |


## [Run #29392287525](https://github.com/sgl-project/sglang/actions/runs/29392287525)
- **分支**: `feat/sm120_glm51`
- **总耗时**: 245.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29392287525

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314544633) |
| stage-b-test-4-npu-a3 | 33.1min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314544668) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业整体失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314544633

- **stage-b-test-4-npu-a3**: 测试套件中5个测试有2个通过，1个失败。失败用例为test_npu_llada2_mini.py，运行870秒后返回退出码1，超过预估时间400秒，可能涉及功能错误或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314544668

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314545698) |
| stage-b-test-1-npu-a2 (1) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314562008) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314567937) |
| multimodal-gen-test-2-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314571900) |
| stage-b-test-2-npu-a2 (0) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314573550) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314573808) |
| stage-b-test-2-npu-a2 (1) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29392287525/job/87314576986) |


## [Run #29350702274](https://github.com/sgl-project/sglang/actions/runs/29350702274)
- **分支**: `jialino/radix-cache-split`
- **总耗时**: 32.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29350702274

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 32.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965875) |
| multimodal-gen-test-1-npu-a3 | 32.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965896) |
| multimodal-gen-test-2-npu-a3 | 32.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965913) |
| stage-b-test-4-npu-a3 | 32.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965940) |
| stage-b-test-1-npu-a2 (0) | 31.8min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965949) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 32.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145968191) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965875

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965896

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965913

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965940

- **stage-b-test-1-npu-a2 (0)**: 日志显示sglang服务正常启动并处理请求，但随后出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965949

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，可能是 CI 配置中引用的模型权重或数据文件未上传到指定存储路径，或路径拼写错误，需检查相关资源是否已正确发布。
  链接: https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145968191

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965961) |
| stage-b-test-2-npu-a2 (0) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145965968) |
| stage-b-test-2-npu-a2 (1) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29350702274/job/87145966023) |


## [Run #29345341583](https://github.com/sgl-project/sglang/actions/runs/29345341583)
- **分支**: `remove-qserve-quantization`
- **总耗时**: 69.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29345341583

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 69.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527679) |
| multimodal-gen-test-2-npu-a3 | 69.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527774) |
| stage-b-test-4-npu-a3 | 69.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527786) |
| multimodal-gen-test-1-npu-a3 | 69.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527803) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 69.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127528392) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527679

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件或数据在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527774

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527786

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527803

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传依赖文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127528392

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527714) |
| stage-b-test-2-npu-a2 (1) | 26.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527726) |
| stage-b-test-1-npu-a2 (1) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527749) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345341583/job/87127527759) |


## [Run #29345312974](https://github.com/sgl-project/sglang/actions/runs/29345312974)
- **分支**: `brayden/fuse-trtllm-gen-prologue-kernels`
- **总耗时**: 77.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29345312974

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411261) |
| multimodal-gen-test-1-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411271) |
| stage-b-test-4-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411278) |
| multimodal-gen-test-2-npu-a3 | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411298) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 77.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411739) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411261

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411271

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411278

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411298

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是配置错误或资源被清理，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411739

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411243) |
| stage-b-test-2-npu-a2 (1) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411290) |
| stage-b-test-1-npu-a2 (0) | 50.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411341) |
| stage-b-test-1-npu-a2 (1) | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29345312974/job/87127411357) |


## [Run #29344715585](https://github.com/sgl-project/sglang/actions/runs/29344715585)
- **分支**: `brayden/remove-aot-router-fused-a-gemm`
- **总耗时**: 77.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29344715585

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 76.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346204) |
| stage-b-test-4-npu-a3 | 76.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346225) |
| stage-b-test-16-npu-a3 | 76.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346237) |
| multimodal-gen-test-2-npu-a3 | 76.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346318) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 76.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346904) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中不存在。这可能是由于 CI 配置引用了错误的 blob 路径，或 blob 已被删除/未上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346204

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346225

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或依赖文件在 Azure Blob 存储中已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346237

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346318

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志文件已被删除或路径错误，属于基础设施/存储问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346904

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346108) |
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346145) |
| stage-b-test-1-npu-a2 (1) | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346147) |
| stage-b-test-2-npu-a2 (1) | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29344715585/job/87125346162) |


## [Run #29342296396](https://github.com/sgl-project/sglang/actions/runs/29342296396)
- **分支**: `bbuf/hpc-ops-attention-backend`
- **总耗时**: 9.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29342296396

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083181) |
| stage-b-test-16-npu-a3 | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083214) |
| stage-b-test-2-npu-a2 (0) | 7.4min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083215) |
| multimodal-gen-test-2-npu-a3 | 8.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083242) |
| stage-b-test-1-npu-a2 (1) | 8.2min | 环境问题 | 自定义容器执行失败，测试进程被中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083268) |
| stage-b-test-4-npu-a3 | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083282) |
| stage-b-test-2-npu-a2 (1) | 7.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083285) |
| stage-b-test-1-npu-a2 (0) | 7.2min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083390) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 8.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083854) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083181

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083214

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行到约7/1319步时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器基础设施问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083215

- **multimodal-gen-test-2-npu-a3**: 错误码BlobNotFound表明作业尝试访问的存储对象缺失，可能是日志上传或下载路径错误，或资源被清理，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083242

- **stage-b-test-1-npu-a2 (1)**: 在运行test_npu_graph_tp1_bf16.py测试时，自定义容器实现执行失败，导致测试进程被终止。日志显示容器在测试开始后约16秒即报错，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083268

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或依赖文件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083282

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试服务已正常启动并响应请求，但随后自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器运行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083285

- **stage-b-test-1-npu-a2 (0)**: 日志显示在测试运行约19秒后，出现错误'Executing the custom container implementation failed'，提示联系自托管runner管理员。这属于runner环境或容器配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083390

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 错误码BlobNotFound表明CI作业尝试下载的依赖文件或数据在存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29342296396/job/87117083854


## [Run #29340140589](https://github.com/sgl-project/sglang/actions/runs/29340140589)
- **分支**: `mmangkad/torch-2.12`
- **总耗时**: 178.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29340140589

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.1min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507629) |
| multimodal-gen-test-1-npu-a3 | 51.5min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507678) |
| stage-b-test-16-npu-a3 | 44.4min | 代码错误 | NPU Deepep 专家并行测试失败，服务启动后测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507781) |

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，5个测试中有4个通过，但test_npu_llada2_mini.py测试失败（退出码1），耗时874秒。该测试属于dllm功能模块，可能是代码逻辑错误或环境配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507629

- **multimodal-gen-test-1-npu-a3**: 作业日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507678

- **stage-b-test-16-npu-a3**: 测试 test_npu_deepep.py 在启动 DeepSeek-R1 模型服务后返回退出码 1，0/1 测试通过，耗时 2415 秒，可能因模型配置或并行策略问题导致服务异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507781

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507608) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507622) |
| multimodal-gen-test-2-npu-a3 | 33.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507696) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507704) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109507720) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29340140589/job/87109508198) |


## [Run #29339745832](https://github.com/sgl-project/sglang/actions/runs/29339745832)
- **分支**: `feat/semantic-radix-backend`
- **总耗时**: 177.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29339745832

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 53.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169408) |
| stage-b-test-4-npu-a3 | 39.1min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169568) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node 20弃用警告和diffusion-failures目录无文件上传提示，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169408

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，5个测试有3个通过，但test_npu_llada2_mini.py失败（退出码1），耗时888秒。该测试属于dllm功能模块，可能涉及LLaDA2模型相关代码问题，需检查该测试的具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169568

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169494) |
| stage-b-test-1-npu-a2 (0) | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169526) |
| stage-b-test-16-npu-a3 | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169575) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169577) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169588) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108169593) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339745832/job/87108170151) |


## [Run #29339195925](https://github.com/sgl-project/sglang/actions/runs/29339195925)
- **分支**: `htphan/fix-symm-mem-cuda-graph-deadlock`
- **总耗时**: 182.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29339195925

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 48.6min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，4/5通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358462) |
| multimodal-gen-test-1-npu-a3 | 56.5min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358612) |

- **stage-b-test-4-npu-a3**: 测试test_npu_llada2_mini.py返回退出码1，耗时906秒，其余4个测试均通过。该测试涉及dllm功能，可能是代码逻辑或环境配置问题导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358462

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有checkout、upload-artifact等步骤，且upload-artifact提示未找到diffusion-failures目录，说明测试可能未运行或提前结束，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358612

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 34.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358373) |
| stage-b-test-16-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358444) |
| stage-b-test-2-npu-a2 (1) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358459) |
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358464) |
| multimodal-gen-test-2-npu-a3 | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358475) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106358514) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339195925/job/87106359220) |


## [Run #29339049315](https://github.com/sgl-project/sglang/actions/runs/29339049315)
- **分支**: `idhanani/unified-radix-swa-fix`
- **总耗时**: 37.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29339049315

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 36.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760586) |
| stage-b-test-16-npu-a3 | 36.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760661) |
| multimodal-gen-test-2-npu-a3 | 36.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760672) |
| stage-b-test-4-npu-a3 | 36.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760750) |
| stage-b-test-1-npu-a2 (0) | 35.9min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760845) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 36.3min | 环境问题 | 日志显示Azure Blob存储返回BlobNotFound错误，CI作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105761298) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或模型权重在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760586

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760661

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760672

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760750

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行正常，但中途出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760845

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 作业日志中仅包含Azure Blob存储的404错误，表明CI在下载或访问依赖文件时失败，文件不存在或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105761298

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760656) |
| stage-b-test-2-npu-a2 (1) | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760668) |
| stage-b-test-1-npu-a2 (1) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29339049315/job/87105760777) |


## [Run #29338838523](https://github.com/sgl-project/sglang/actions/runs/29338838523)
- **分支**: `bbuf/hpc-ops-attention-backend`
- **总耗时**: 36.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29338838523

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 35.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089452) |
| stage-b-test-4-npu-a3 | 35.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089491) |
| multimodal-gen-test-1-npu-a3 | 35.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089499) |
| stage-b-test-1-npu-a2 (0) | 35.0min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089514) |
| multimodal-gen-test-2-npu-a3 | 35.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089544) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 35.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105090117) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089452

- **stage-b-test-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或数据在 Azure Blob 中已被删除或路径错误，可能是上游作业未成功上传或存储配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089491

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089499

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到51%时出现"Executing the custom container implementation failed"错误，属于自托管runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089514

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089544

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是文件被清理、路径错误或上传失败，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105090117

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089494) |
| stage-b-test-1-npu-a2 (1) | 32.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089504) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338838523/job/87105089559) |


## [Run #29338486836](https://github.com/sgl-project/sglang/actions/runs/29338486836)
- **分支**: `pr_add_multi_stream_gemm_fusion`
- **总耗时**: 172.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29338486836

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.7min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，4/5通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825432) |
| multimodal-gen-test-1-npu-a3 | 52.5min | 其他 | 日志不完整，未显示测试失败的具体原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825457) |

- **stage-b-test-4-npu-a3**: 测试test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py执行失败（exit code 1），耗时915秒，其余4个NPU测试均通过，表明该测试用例本身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825432

- **multimodal-gen-test-1-npu-a3**: 日志仅包含作业启动、checkout和upload-artifact步骤，未展示multimodal-gen测试的实际执行结果或错误信息。测试可能因环境问题、代码错误或超时失败，但日志中无相关线索。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825457

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825389) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825413) |
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825429) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825461) |
| multimodal-gen-test-2-npu-a3 | 37.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825467) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825504) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338486836/job/87103825904) |


## [Run #29338080660](https://github.com/sgl-project/sglang/actions/runs/29338080660)
- **分支**: `kernels/phase25-vendored`
- **总耗时**: 172.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29338080660

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.6min | 代码错误 | 测试 test_npu_llada2_mini.py 失败，返回退出码 1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463204) |
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463285) |

- **stage-b-test-4-npu-a3**: 在 NPU A3 环境下，5 个测试中 4 个通过，仅 test_npu_llada2_mini.py 失败（耗时 870 秒），具体错误信息未在日志中显示，需进一步查看该测试的详细输出以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463204

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463285

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463067) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463096) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463162) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463183) |
| stage-b-test-2-npu-a2 (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463203) |
| multimodal-gen-test-2-npu-a3 | 51.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463260) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29338080660/job/87102463737) |


## [Run #29337703742](https://github.com/sgl-project/sglang/actions/runs/29337703742)
- **分支**: `kernels/phase25-dsa-dsv4`
- **总耗时**: 176.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29337703742

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.0min | 代码错误 | 测试 test_npu_llada2_mini.py 失败，返回退出码 1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137833) |
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137846) |

- **stage-b-test-4-npu-a3**: 在 NPU A3 环境下，5 个测试中 4 个通过，仅 test_npu_llada2_mini.py 失败（耗时 878 秒），具体错误信息未在日志中显示，需进一步查看该测试的详细输出以定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137833

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、checkout和upload-artifact步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137846

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137829) |
| multimodal-gen-test-2-npu-a3 | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137844) |
| stage-b-test-1-npu-a2 (0) | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137850) |
| stage-b-test-2-npu-a2 (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137859) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137874) |
| stage-b-test-16-npu-a3 | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101137898) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337703742/job/87101138266) |


## [Run #29337380180](https://github.com/sgl-project/sglang/actions/runs/29337380180)
- **分支**: `fuse-gate-gemv-into-append`
- **总耗时**: 153.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29337380180

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 52.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038638) |
| stage-b-test-4-npu-a3 | 47.0min | 代码错误 | 测试 test_npu_llada2_mini.py 失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038673) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但真正失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038638

- **stage-b-test-4-npu-a3**: 在NPU A3环境下，5个测试中4个通过，仅test_npu_llada2_mini.py失败，耗时878秒，未显示具体错误信息，可能为代码逻辑或环境兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038673

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 47.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038663) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038672) |
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038708) |
| stage-b-test-1-npu-a2 (1) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038724) |
| stage-b-test-2-npu-a2 (1) | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038745) |
| stage-b-test-16-npu-a3 | 17.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100038763) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29337380180/job/87100039309) |


## [Run #29330274051](https://github.com/sgl-project/sglang/actions/runs/29330274051)
- **分支**: `feat/lora-merge-ipc-update`
- **总耗时**: 162.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29330274051

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 57.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29330274051/job/87076364690) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到diffusion-failures目录。实际失败原因需查看完整日志或测试输出。
  链接: https://github.com/sgl-project/sglang/actions/runs/29330274051/job/87076364690

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29330274051/job/87076364647) |


## [Run #29329839778](https://github.com/sgl-project/sglang/actions/runs/29329839778)
- **分支**: `fix/glm-tool-parser-escapes`
- **总耗时**: 171.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29329839778

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.9min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930348) |
| multimodal-gen-test-1-npu-a3 | 62.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930350) |
| stage-b-test-2-npu-a2 (1) | 9.3min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930351) |
| stage-b-test-2-npu-a2 (0) | 4.0min | 环境问题 | pip下载依赖包时网络连接中断，导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930378) |

- **stage-b-test-4-npu-a3**: 测试套件中4/5通过，但test_npu_llada2_mini.py返回退出码1，耗时903秒，可能因代码逻辑错误或环境问题导致该用例失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930348

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions环境初始化、checkout和artifact上传步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930350

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行到17%时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是NPU环境或容器问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930351

- **stage-b-test-2-npu-a2 (0)**: 在安装Python依赖过程中，pip从远程下载包时出现IncompleteRead错误，连接中断导致下载不完整，最终安装失败。属于网络不稳定或镜像源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930378

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930302) |
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930323) |
| multimodal-gen-test-2-npu-a3 | 39.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930384) |
| stage-b-test-1-npu-a2 (1) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930407) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29329839778/job/87074930703) |


## [Run #29328209895](https://github.com/sgl-project/sglang/actions/runs/29328209895)
- **分支**: `amd_fix_deepseekv4_0714`
- **总耗时**: 167.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29328209895

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.7min | 其他 | 测试用例 test_npu_llada2_mini.py 失败，但日志未显示具体错误原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103353299) |

- **stage-b-test-4-npu-a3**: 作业中4/5测试通过，仅 test_npu_llada2_mini.py 返回退出码1，耗时884秒。日志被截断，未包含该用例的具体失败信息，无法判断是代码错误、环境问题还是其他原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103353299

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354020) |
| stage-b-test-2-npu-a2 (1) | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354095) |
| stage-b-test-1-npu-a2 (0) | 41.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354195) |
| stage-b-test-1-npu-a2 (1) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354466) |
| stage-b-test-2-npu-a2 (0) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354467) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29328209895/job/87103354651) |


## [Run #29322097405](https://github.com/sgl-project/sglang/actions/runs/29322097405)
- **分支**: `main`
- **总耗时**: 139.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29322097405

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 138.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683822) |
| multimodal-gen-test-1-npu-a3 | 21.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683835) |
| multimodal-gen-test-2-npu-a3 | 138.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683851) |
| stage-b-test-2-npu-a2 (0) | 4.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683856) |
| stage-b-test-4-npu-a3 | 33.0min | 代码错误 | 测试用例 test_npu_llada2_mini.py 执行失败，退出码为 1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683894) |
| stage-b-test-2-npu-a2 (1) | 3.5min | 环境问题 | pip下载依赖时网络连接中断，导致安装失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683933) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.3min | 其他 | 作业日志显示测试状态为pass，但作业被标记为失败，可能是基础设施或清理阶段问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049684043) |

- **stage-b-test-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683822

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683835

- **multimodal-gen-test-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683851

- **stage-b-test-2-npu-a2 (0)**: 作业在运行测试命令后，自定义容器实现执行失败，提示联系自托管runner管理员，属于基础设施环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683856

- **stage-b-test-4-npu-a3**: 在 stage-b-test-4-npu-a3 作业中，测试 test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py 失败，退出码为 1，导致整体测试 2/5 通过。该测试用例本身存在代码错误或断言失败，需检查该测试脚本的具体逻辑。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683894

- **stage-b-test-2-npu-a2 (1)**: 日志显示pip在下载包时出现IncompleteRead错误，仅读取了17MB但预期还有170MB，网络连接中断导致依赖安装失败，属于环境网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683933

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志中test_status为pass，测试本身通过。失败可能源于后续的plog备份或清理阶段，但日志未显示明确错误，仅见Node.js 20弃用警告，需进一步查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049684043

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (1) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683837) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29322097405/job/87049683862) |


## [Run #29321466158](https://github.com/sgl-project/sglang/actions/runs/29321466158)
- **分支**: `kernels/phase25-linear`
- **总耗时**: 141.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29321466158

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 46.6min | 代码错误 | NPU测试test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612713) |
| multimodal-gen-test-1-npu-a3 | 52.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612720) |

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，测试test_npu_llada2_mini.py运行失败（exit code 1），其余4个测试均通过。该测试属于dllm功能模块，可能因代码逻辑错误或环境依赖问题导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612713

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、checkout、上传artifact等常规步骤，未显示任何测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612720

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612722) |
| stage-b-test-1-npu-a2 (1) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612759) |
| stage-b-test-1-npu-a2 (0) | 43.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612775) |
| stage-b-test-2-npu-a2 (1) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612776) |
| multimodal-gen-test-2-npu-a3 | 35.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612787) |
| stage-b-test-2-npu-a2 (0) | 17.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047612795) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321466158/job/87047613145) |


## [Run #29321089757](https://github.com/sgl-project/sglang/actions/runs/29321089757)
- **分支**: `codex/support-fal-ideogram-v4-fast-instant`
- **总耗时**: 140.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29321089757

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 53.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798879) |
| stage-b-test-4-npu-a3 | 47.6min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798898) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含Node 20弃用警告、上传diffusion-failures目录时未找到文件等提示，未展示测试执行过程或具体失败断言，无法判断失败根因，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798879

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，测试文件test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py运行失败（exit code 1），其余4个测试均通过。该测试用例本身存在代码或环境适配问题，导致整体作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798898

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798865) |
| stage-b-test-1-npu-a2 (0) | 42.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798875) |
| stage-b-test-1-npu-a2 (1) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798888) |
| stage-b-test-2-npu-a2 (1) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798901) |
| stage-b-test-2-npu-a2 (0) | 16.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798907) |
| stage-b-test-16-npu-a3 | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046798922) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29321089757/job/87046799478) |


## [Run #29319625597](https://github.com/sgl-project/sglang/actions/runs/29319625597)
- **分支**: `cctry/kv-to-page-indices-on-device`
- **总耗时**: 151.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29319625597

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 33.6min | 代码错误 | NPU测试中test_npu_llada2_mini.py失败，导致作业整体退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655243) |
| multimodal-gen-test-1-npu-a3 | 51.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境警告和上传产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655282) |

- **stage-b-test-4-npu-a3**: 测试摘要显示2/5通过，失败项为test_npu_llada2_mini.py（退出码1），其余两个测试通过。失败原因可能是该测试用例存在代码逻辑或环境依赖问题，需进一步查看具体报错。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655243

- **multimodal-gen-test-1-npu-a3**: 日志截断，缺少测试执行和失败断言部分。仅见Node 20弃用警告及diffusion-failures目录无文件上传提示，无法定位具体失败点，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655282

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655259) |
| multimodal-gen-test-2-npu-a3 | 35.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655270) |
| stage-b-test-16-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655295) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655300) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655301) |
| stage-b-test-1-npu-a2 (1) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041655314) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319625597/job/87041656019) |


## [Run #29319427123](https://github.com/sgl-project/sglang/actions/runs/29319427123)
- **分支**: `pr_add_multi_stream_gemm_fusion`
- **总耗时**: 152.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29319427123

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-1-npu-a2 (1) | 3.2min | 环境问题 | pip下载依赖时网络连接中断，导致安装失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021057) |
| multimodal-gen-test-1-npu-a3 | 58.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021059) |
| stage-b-test-4-npu-a3 | 46.9min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021085) |

- **stage-b-test-1-npu-a2 (1)**: 在安装Python依赖包时，pip从网络下载文件过程中出现IncompleteRead错误，连接被中断，导致安装流程失败，属于网络环境不稳定问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021057

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021059

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，测试test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py运行失败（exit code 1），其余4个测试均通过。该测试用例存在代码或环境相关问题，导致整体作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021085

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 42.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021080) |
| multimodal-gen-test-2-npu-a3 | 34.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021102) |
| stage-b-test-2-npu-a2 (1) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021106) |
| stage-b-test-16-npu-a3 | 18.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021115) |
| stage-b-test-2-npu-a2 (0) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021134) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29319427123/job/87041021329) |


## [Run #29318774860](https://github.com/sgl-project/sglang/actions/runs/29318774860)
- **分支**: `glm52/mtp-split-1-topk1`
- **总耗时**: 174.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29318774860

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 47.6min | 代码错误 | NPU测试test_npu_llada2_mini.py失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935038) |
| multimodal-gen-test-1-npu-a3 | 62.8min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935048) |

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，5个NPU测试有4个通过，仅test_npu_llada2_mini.py失败（耗时868秒），返回退出码1，其余测试均正常，判断为该测试用例本身存在代码或环境适配问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935038

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但实际失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935048

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (1) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935095) |
| stage-b-test-2-npu-a2 (0) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935195) |
| stage-b-test-1-npu-a2 (1) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935867) |
| multimodal-gen-test-2-npu-a3 | 41.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935874) |
| stage-b-test-1-npu-a2 (0) | 42.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100935904) |
| stage-b-test-16-npu-a3 | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100936291) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29318774860/job/87100936568) |


## [Run #29289186291](https://github.com/sgl-project/sglang/actions/runs/29289186291)
- **分支**: `glm5/moe-output-output`
- **总耗时**: 249.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29289186291

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 33.4min | 超时 | 测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585300) |
| multimodal-gen-test-1-npu-a3 | 62.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585316) |

- **stage-b-test-4-npu-a3**: test_npu_llada2_mini.py 运行超过900秒（预估400秒），超时被强制终止，返回退出码1，导致作业失败。其余4个测试中2个通过，2个未运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585300

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有checkout、upload-artifact等步骤，且upload-artifact提示无文件上传。可能测试未运行或日志被截断，需查看完整日志确认失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585316

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-1-npu-a2 (0) | 41.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315585994) |
| stage-b-test-2-npu-a2 (0) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586091) |
| stage-b-test-1-npu-a2 (1) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586296) |
| multimodal-gen-test-2-npu-a3 | 45.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586354) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586460) |
| stage-b-test-16-npu-a3 | 18.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586527) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29289186291/job/87315586917) |


## [Run #29202367445](https://github.com/sgl-project/sglang/actions/runs/29202367445)
- **分支**: `dp-attn-free-port-block`
- **总耗时**: 60.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29202367445

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 35.4min | 代码错误 | 测试用例 test_npu_llada2_mini.py 执行失败，退出码为1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819978) |
| multimodal-gen-test-1-npu-a3 | 58.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819982) |

- **stage-b-test-4-npu-a3**: 在 stage-b-test-4-npu-a3 作业中，测试文件 test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py 运行失败（exit code 1），其余4个测试均通过。该测试耗时仅171秒，可能因代码逻辑错误或环境问题导致失败，需进一步查看具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819978

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node.js弃用警告及上传artifact时未找到diffusion-failures目录，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819982

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-16-npu-a3 | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819969) |
| multimodal-gen-test-2-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819973) |
| stage-b-test-1-npu-a2 (1) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819981) |
| stage-b-test-2-npu-a2 (1) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819983) |
| stage-b-test-2-npu-a2 (0) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819984) |
| stage-b-test-1-npu-a2 (0) | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675819985) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29202367445/job/86675820186) |


## [Run #29201454263](https://github.com/sgl-project/sglang/actions/runs/29201454263)
- **分支**: `jit-dtype-trait-reduce-fix`
- **总耗时**: 26.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29201454263

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 25.5min | 环境问题 | 自定义容器执行失败，runner环境异常导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678065954) |

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常（进度82%），但突然报错"Executing the custom container implementation failed"，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678065954

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066155) |
| stage-b-test-1-npu-a2 (0) | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066284) |
| stage-b-test-2-npu-a2 (1) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066297) |
| stage-b-test-1-npu-a2 (1) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066302) |
| stage-b-test-16-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066339) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201454263/job/86678066498) |


## [Run #29201063345](https://github.com/sgl-project/sglang/actions/runs/29201063345)
- **分支**: `jit_dsv4_c128_opt`
- **总耗时**: 42.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29201063345

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 36.1min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442393) |

- **stage-b-test-4-npu-a3**: 在stage-b-test-4-npu-a3作业中，测试test_npu_llada2_mini.py运行190秒后失败（exit code 1），其余4个NPU测试均通过。该测试属于dllm功能模块，可能涉及LLaDA2模型相关代码问题，需查看具体错误日志定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442393

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| stage-b-test-2-npu-a2 (0) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442407) |
| stage-b-test-16-npu-a3 | 16.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442412) |
| stage-b-test-1-npu-a2 (0) | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442417) |
| stage-b-test-1-npu-a2 (1) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442440) |
| stage-b-test-2-npu-a2 (1) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442463) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29201063345/job/86672442620) |


## [Run #29199118191](https://github.com/sgl-project/sglang/actions/runs/29199118191)
- **分支**: `tom_refactor_202605a/primary/nonmech_model_runner`
- **总耗时**: 12.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29199118191

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-4-npu-a3 | 3.7min | 代码错误 | NPU W4A4量化测试失败，测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336072) |
| stage-b-test-2-npu-a2 (0) | 5.3min | 代码错误 | NPU图模式TP2 BF16测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336076) |
| stage-b-test-16-npu-a3 | 4.1min | 代码错误 | NPU DeepEP测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336080) |
| stage-b-test-1-npu-a2 (1) | 5.1min | 代码错误 | NPU采样后端测试失败，测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336081) |
| stage-b-test-1-npu-a2 (0) | 5.1min | 环境问题 | NPU测试用例test_npu_hicache_mha.py执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336086) |
| multimodal-gen-test-2-npu-a3 | 7.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336088) |
| stage-b-test-2-npu-a2 (1) | 5.0min | 代码错误 | NPU MLA FIA W8A8 INT8 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336090) |
| multimodal-gen-test-1-npu-a3 | 11.6min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336103) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.8min | 其他 | 日志被截断，未显示实际测试结果，无法确定失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336237) |

- **stage-b-test-4-npu-a3**: test_npu_w4a4_quantization.py测试执行失败，退出码为1，导致整个作业失败。可能是量化实现或测试用例本身存在问题，需检查该测试的具体报错信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336072

- **stage-b-test-2-npu-a2 (0)**: 测试文件test_npu_graph_tp2_bf16.py执行失败，0/2测试通过，耗时74秒。可能是NPU图模式相关代码存在bug或环境配置问题，需查看具体测试输出定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336076

- **stage-b-test-16-npu-a3**: test_npu_deepep.py测试在NPU A3环境下运行55.59秒后失败，退出码为1。该测试属于专家并行策略测试，可能涉及DeepEP通信库的NPU适配问题，需要查看具体错误日志定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336080

- **stage-b-test-1-npu-a2 (1)**: 测试文件test_npu_sampling_backend.py执行失败，退出码为1，导致整个作业失败。日志中未显示具体错误信息，但测试摘要显示0/4通过，可能是测试代码或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336081

- **stage-b-test-1-npu-a2 (0)**: 测试文件test/registered/ascend/basic_function/HiCache/test_npu_hicache_mha.py运行失败，退出码为1，导致整个作业失败。日志显示测试在74秒内结束，但未通过，可能是环境配置或模型加载问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336086

- **multimodal-gen-test-2-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅显示上传diffusion-failures工件时未找到文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336088

- **stage-b-test-2-npu-a2 (1)**: 测试文件 test_npu_mla_fia_w8a8int8.py 执行失败，0/2 测试通过，耗时74秒，未超时。可能是代码逻辑错误或NPU环境兼容性问题，需查看具体测试输出定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336090

- **multimodal-gen-test-1-npu-a3**: 作业在运行multimodal-gen测试后尝试上传diffusion-failures目录，但未找到任何文件，说明测试可能未产生失败样本或测试未执行。日志中间部分被省略，无法定位具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336103

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志仅包含作业初始化和清理阶段，未包含测试执行及失败的具体错误信息，因此无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29199118191/job/86667336237


## [Run #29197701797](https://github.com/sgl-project/sglang/actions/runs/29197701797)
- **分支**: `tom_refactor_202605a/primary/nonmech_model_runner`
- **总耗时**: 44.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29197701797

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 4.3min | 代码错误 | NPU DeepEP 测试失败，测试用例返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587419) |
| multimodal-gen-test-2-npu-a3 | 42.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587420) |
| stage-b-test-4-npu-a3 | 3.7min | 代码错误 | NPU MLA W8A8INT8 测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587431) |
| multimodal-gen-test-1-npu-a3 | 42.8min | 其他 | 日志不完整，未显示测试失败的具体原因，仅包含环境警告和上传工件信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587433) |
| stage-b-test-2-npu-a2 (0) | 5.2min | 代码错误 | NPU图模式TP2 BF16测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587438) |
| stage-b-test-1-npu-a2 (1) | 5.2min | 代码错误 | NPU采样后端测试失败，测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587440) |
| stage-b-test-2-npu-a2 (1) | 5.2min | 代码错误 | NPU测试用例test_npu_mla_fia_w8a8int8.py执行失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587450) |
| stage-b-test-1-npu-a2 (0) | 5.3min | 代码错误 | HiCache MHA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587458) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.8min | 环境问题 | 作业在启动阶段即被终止，未进入实际测试执行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587730) |

- **stage-b-test-16-npu-a3**: test_npu_deepep.py 测试在 expert_parallelism 策略下执行失败，0/1 测试通过，耗时约55秒，具体失败原因需查看测试日志中的断言或异常信息。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587419

- **multimodal-gen-test-2-npu-a3**: 日志仅包含环境准备和清理过程，未展示测试执行及失败详情。上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物或提前退出，需查看完整日志定位具体失败点。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587420

- **stage-b-test-4-npu-a3**: 测试文件 test_npu_mla_w8a8int8.py 在运行45秒后失败，0/5测试通过。可能是该测试用例存在代码逻辑错误或与当前NPU环境不兼容，导致运行时异常退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587431

- **multimodal-gen-test-1-npu-a3**: 日志被截断，缺少测试执行和失败的关键部分。仅看到Node 20弃用警告、上传diffusion-failures工件时未找到文件等非致命信息，无法判断实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587433

- **stage-b-test-2-npu-a2 (0)**: 测试文件test_npu_graph_tp2_bf16.py执行失败，0/2测试通过，耗时74秒。可能是NPU图模式相关代码存在bug或环境配置问题，需查看具体测试输出定位错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587438

- **stage-b-test-1-npu-a2 (1)**: test_npu_sampling_backend.py测试执行失败，退出码为1，导致整个作业失败。日志中未显示具体错误信息，但测试文件本身存在问题，可能是代码逻辑错误或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587440

- **stage-b-test-2-npu-a2 (1)**: 测试文件test/registered/ascend/basic_function/runtime_opts/test_npu_mla_fia_w8a8int8.py在运行过程中返回退出码1，导致整个作业失败。具体错误信息未在日志中详细展示，但可判断为该测试用例本身存在问题，可能是代码逻辑错误或环境配置不满足要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587450

- **stage-b-test-1-npu-a2 (0)**: 测试文件 test_npu_hicache_mha.py 执行失败，0/5 测试通过，耗时74秒，具体错误信息未在日志中显示，可能是测试断言失败或运行时错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587458

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试脚本，可能因runner环境初始化失败或资源分配问题导致作业提前结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/29197701797/job/86663587730


## [Run #29193506426](https://github.com/sgl-project/sglang/actions/runs/29193506426)
- **分支**: `tom_refactor_202605a/primary/nonmech_model_runner`
- **总耗时**: 56.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29193506426

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 4.2min | 代码错误 | NPU DeepEP专家并行测试失败，测试用例返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277069) |
| multimodal-gen-test-1-npu-a3 | 55.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277076) |
| stage-b-test-2-npu-a2 (1) | 5.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277084) |
| stage-b-test-4-npu-a3 | 3.8min | 环境问题 | NPU测试用例test_npu_w4a4_quantization.py执行失败，退出码1，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277089) |
| stage-b-test-1-npu-a2 (1) | 5.2min | 代码错误 | NPU采样后端测试失败，测试文件返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277093) |
| stage-b-test-1-npu-a2 (0) | 5.3min | 代码错误 | HiCache MHA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277105) |
| stage-b-test-2-npu-a2 (0) | 5.1min | 代码错误 | NPU图模式TP2 BF16测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277129) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 3.7min | 其他 | 日志被截断，无法确定具体失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277252) |

- **stage-b-test-16-npu-a3**: 测试文件test_npu_deepep.py执行失败，退出码为1，导致整个作业终止。可能是代码逻辑错误或环境配置问题，需查看具体测试输出定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277069

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，只有GitHub Actions的常规准备、上传artifact（无文件）和清理步骤。无法判断测试是否失败或失败原因，可能日志被截断或测试未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277076

- **stage-b-test-2-npu-a2 (1)**: 作业在启动NPU推理服务时，自定义容器实现执行失败，日志显示runner无法正常完成容器操作，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277084

- **stage-b-test-4-npu-a3**: 测试文件test/registered/ascend/basic_function/quant/test_npu_w4a4_quantization.py在NPU A3环境下运行失败，退出码为1，测试摘要显示0/5通过。可能是环境配置或依赖问题，需进一步查看该测试的具体错误日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277089

- **stage-b-test-1-npu-a2 (1)**: 测试文件test_npu_sampling_backend.py执行失败，退出码1，导致整个作业失败。可能是测试逻辑或环境配置问题，需查看具体测试输出定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277093

- **stage-b-test-1-npu-a2 (0)**: 测试文件test_npu_hicache_mha.py在NPU A2环境下运行失败，耗时73秒，0/5测试通过。可能是测试代码逻辑错误或与当前NPU环境不兼容导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277105

- **stage-b-test-2-npu-a2 (0)**: 测试文件test_npu_graph_tp2_bf16.py执行失败，0/2测试通过，耗时74秒。可能是NPU图模式相关代码存在bug或环境配置问题，需查看具体测试输出定位错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277129

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 提供的日志仅包含作业启动和清理阶段，未显示测试执行或失败的具体错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277252

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-2-npu-a3 | 37.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/29193506426/job/86652277133) |


## [Run #29193175682](https://github.com/sgl-project/sglang/actions/runs/29193175682)
- **分支**: `main`
- **总耗时**: 10.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/29193175682

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| stage-b-test-16-npu-a3 | 9.3min | 环境问题 | NPU作业在模型权重加载阶段出现Scheduler watchdog超时，导致自定义容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366892) |
| stage-b-test-4-npu-a3 | 9.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366895) |
| stage-b-test-1-npu-a2 (1) | 9.2min | 环境问题 | 自定义容器执行失败，NPU后端算子回退导致服务异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366903) |
| multimodal-gen-test-1-npu-a3 | 9.2min | 其他 | 作业正常结束，无失败迹象，仅上传工件时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366910) |
| multimodal-gen-test-2-npu-a3 | 9.3min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366911) |
| stage-b-test-2-npu-a2 (0) | 9.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366918) |
| stage-b-test-1-npu-a2 (0) | 9.2min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366930) |
| stage-b-test-2-npu-a2 (1) | 9.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366936) |
| single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms | 5.8min | 环境问题 | 作业在启动后立即失败，未执行实际测试，可能因运行环境或资源问题导致。 | [job link](https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651367037) |

- **stage-b-test-16-npu-a3**: 日志显示在加载MoE模型权重时（约78%进度）出现Scheduler watchdog timeout，随后自定义容器实现执行失败。这可能是NPU环境资源紧张或驱动问题导致加载缓慢，而非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366892

- **stage-b-test-4-npu-a3**: 日志显示测试运行正常，但在12:54:54时出现"Executing the custom container implementation failed"错误，随后作业清理退出。这属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366895

- **stage-b-test-1-npu-a2 (1)**: 日志显示NPU后端不支持aten::_assert_async算子，回退到CPU执行，随后自定义容器实现执行失败，导致作业终止。可能是NPU环境配置或算子兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366903

- **multimodal-gen-test-1-npu-a3**: 日志显示作业成功完成，仅在上传diffusion-failures工件时提示无文件，属正常情况。未发现测试失败或错误，可能为作业提前退出或测试未执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366910

- **multimodal-gen-test-2-npu-a3**: 作业在运行后上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本或提前退出。但关键测试日志被省略，无法定位具体失败点，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366911

- **stage-b-test-2-npu-a2 (0)**: 日志显示测试运行正常，但在执行过程中出现错误："Executing the custom container implementation failed"，提示联系自托管 runner 管理员，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366918

- **stage-b-test-1-npu-a2 (0)**: 日志显示测试运行到58%时出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366930

- **stage-b-test-2-npu-a2 (1)**: 日志显示测试运行正常（吞吐量2826 token/s），但在12:54:55时出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651366936

- **single-node-poc (qwen3_6_27b_w8a8_1p_in64k_out1k_50ms, linux-aarch64-a3-2, test/registered/ascend... / qwen3_6_27b_w8a8_1p_in64k_out1k_50ms**: 日志显示作业在准备阶段后直接进入清理流程，未运行测试脚本，且未生成metrics.json，可能因NPU资源分配失败或环境初始化异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/29193175682/job/86651367037


## [Run #31727270352](https://github.com/sgl-project/sglang/actions/runs/31727270352)
- **分支**: `idhanani/unified-radix-swa-fix`
- **总耗时**: 198.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31727270352

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538893255) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94571191108) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 26.2min | 性能回归 | NPU性能测试未通过，0/4测试全部失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94577335536) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.7min | 其他 | 健康检查快速失败，因同PR中另一作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94583648906) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，属于runner容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538893255

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1108秒后退出码1，性能指标未达标，导致整个perf测试套件失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94571191108

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: qwen3_235b_a22b模型的w8a8_8p_in3k5_out1k5_50ms性能测试失败，退出码1，耗时1213秒超过预估3600秒限制，可能因性能未达标或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94577335536

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94583648906

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 35.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538892726) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538892749) |
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538892784) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538892892) |
| base-b-test-16-npu-a3 / run (0) | 67.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538892896) |
| base-b-test-4-npu-a3 / run (0) | 31.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538893016) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538893017) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538893135) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538893169) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538893308) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727270352/job/94538893775) |


## [Run #31727235228](https://github.com/sgl-project/sglang/actions/runs/31727235228)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 21.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31727235228

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 20.2min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688236) |
| base-b-test-16-npu-a3 / run (0) | 20.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688430) |
| base-b-test-2-npu-a3 / run (0) | 20.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688499) |
| base-b-test-1-npu-a3 / run (0) | 20.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688521) |
| base-b-test-8-npu-a3 / run (0) | 20.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688538) |
| base-b-test-4-npu-a3 / run (1) | 20.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688553) |
| base-b-test-4-npu-a3 / run (0) | 20.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688701) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688919) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538689042) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538689051) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 20.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538689061) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688236

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 资源已被删除或路径错误，属于基础设施/存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688430

- **base-b-test-2-npu-a3 / run (0)**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688499

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688521

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是上游产物未上传或过期，属于环境/资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688538

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688553

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688701

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被清理，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688919

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的存储位置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538689042

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失，可能是上传失败或路径错误，需检查资源是否存在及访问权限。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538689051

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期清理或配置变更所致，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538689061

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727235228/job/94538688498) |


## [Run #31727209003](https://github.com/sgl-project/sglang/actions/runs/31727209003)
- **分支**: `main`
- **总耗时**: 102.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31727209003

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538720920) |
| base-b-test-16-npu-a3 / run (0) | 101.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538720979) |
| base-b-test-8-npu-a3 / run (0) | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538720983) |
| base-b-test-4-npu-a3 / run (0) | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721092) |
| base-b-test-4-npu-a3 / run (1) | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721230) |
| base-b-test-1-npu-a3 / run (0) | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721288) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721453) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721575) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721643) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 101.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721702) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境/资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538720920

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538720979

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538720983

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是缓存或依赖文件未正确上传，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721092

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721230

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721288

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721453

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721575

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721643

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储对象缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538721702

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727209003/job/94538720894) |


## [Run #31727027724](https://github.com/sgl-project/sglang/actions/runs/31727027724)
- **分支**: `agent/whisper-long-audio-chunking`
- **总耗时**: 189.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31727027724

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94569001183) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94576683967) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94578675776) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94589567971) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1117秒后返回退出码1，0/1测试通过，属于性能测试未达标或执行错误，非环境或超时问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94569001183

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因作业，因此本作业（base-c-test-perf-16-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94576683967

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，作为根因作业，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94578675776

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94589567971

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537968732) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537968908) |
| base-b-test-16-npu-a3 / run (0) | 51.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537968948) |
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537968970) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537968994) |
| base-b-test-4-npu-a3 / run (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537969062) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537969082) |
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537969140) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537969493) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537969564) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537969592) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31727027724/job/94537969650) |


## [Run #31726053280](https://github.com/sgl-project/sglang/actions/runs/31726053280)
- **分支**: `idhanani/unified-radix-swa-fix`
- **总耗时**: 15.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31726053280

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534676765) |
| base-b-test-2-npu-a3 / run (0) | 14.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534676924) |
| base-b-test-16-npu-a3 / run (0) | 14.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534676935) |
| base-b-test-1-npu-a3 / run (0) | 14.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677024) |
| base-b-test-8-npu-a3 / run (0) | 14.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677058) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677147) |
| base-b-test-4-npu-a3 / run (0) | 14.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677176) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677487) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677530) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677594) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 14.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677602) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未生成失败产物，但根本原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534676765

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成产物，或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534676924

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534676935

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677024

- **base-b-test-8-npu-a3 / run (0)**: 作业在下载或访问某个blob时，返回BlobNotFound错误，可能是由于blob被删除、路径错误或存储账户配置问题，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677058

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677147

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是缓存失效或资源清理导致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677176

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677487

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677530

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677594

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534677602

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31726053280/job/94534676978) |


## [Run #31725107558](https://github.com/sgl-project/sglang/actions/runs/31725107558)
- **分支**: `feat/kv-events-component-placement-v2`
- **总耗时**: 193.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31725107558

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 72.9min | 精度回归 | NPU精度测试用例qwen3_5_9b_bf16_1p_gsm8k执行失败，0/3测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545749) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94569676616) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94574837706) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94575083733) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py运行4160秒后返回退出码1，所有3个测试均未通过，表明模型精度或推理结果不符合预期，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545749

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1134秒后退出码1，属于性能测试未通过，可能因模型推理速度未达预期或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94569676616

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，但根因作业失败导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94574837706

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业在启动后，健康检查发现同一次运行中另一个作业（base-c-test-perf-8-npu-a3）已失败，触发了fast-fail逻辑，导致本作业被跳过并报错退出，并非本作业自身存在问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94575083733

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531544859) |
| base-b-test-8-npu-a3 / run (0) | 12.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545137) |
| base-b-test-2-npu-a3 / run (0) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545173) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545194) |
| base-b-test-4-npu-a3 / run (1) | 10.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545210) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545247) |
| base-b-test-16-npu-a3 / run (0) | 49.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545348) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545358) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 10.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545669) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545676) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31725107558/job/94531545742) |


## [Run #31724431325](https://github.com/sgl-project/sglang/actions/runs/31724431325)
- **分支**: `xpu-dcp-support`
- **总耗时**: 255.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31724431325

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.2min | 超时 | 性能测试用例执行超时或失败，导致作业整体退出。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94562765706) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94571772778) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94572927445) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94597480322) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1065秒后返回退出码1，未通过，导致作业失败。可能因性能未达标或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94562765706

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因失败，导致本作业（base-c-test-perf-4-npu-a3）被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94571772778

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业在健康检查阶段因检测到其他作业（base-c-test-perf-8-npu-a3）失败而触发快速失败机制，本作业本身未开始执行测试，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94572927445

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业，导致本作业被快速失败（fast-fail）跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94597480322

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529301585) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529301592) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529301604) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529301624) |
| base-b-test-4-npu-a3 / run (0) | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529301681) |
| base-b-test-2-npu-a3 / run (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529301689) |
| base-b-test-16-npu-a3 / run (0) | 47.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529301692) |
| base-b-test-4-npu-a3 / run (1) | 10.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529301727) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529302027) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529302085) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 126.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529302147) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724431325/job/94529302160) |


## [Run #31724230737](https://github.com/sgl-project/sglang/actions/runs/31724230737)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 36.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31724230737

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692514) |
| base-b-test-16-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692524) |
| base-b-test-2-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692592) |
| base-b-test-4-npu-a3 / run (1) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692670) |
| base-b-test-1-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692733) |
| base-b-test-4-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692936) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528693044) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528693102) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528693165) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528693184) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692514

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误所致，属于基础设施或环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692524

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692592

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692670

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是缓存失效或资源清理导致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692733

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692936

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528693044

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到，可能是文件被删除、路径错误或上传失败，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528693102

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528693165

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528693184

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692474) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724230737/job/94528692548) |


## [Run #31724189348](https://github.com/sgl-project/sglang/actions/runs/31724189348)
- **分支**: `main`
- **总耗时**: 36.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31724189348

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547566) |
| base-b-test-16-npu-a3 / run (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547682) |
| base-b-test-1-npu-a3 / run (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547729) |
| base-b-test-4-npu-a3 / run (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547730) |
| base-b-test-2-npu-a3 / run (0) | 35.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547761) |
| base-b-test-8-npu-a3 / run (0) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547783) |
| base-b-test-4-npu-a3 / run (1) | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547826) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528548014) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528548045) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528548064) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528548150) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547566

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547682

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547729

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547730

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象缺失，可能是构建产物未上传、路径错误或存储被清理，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547761

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547783

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的构建产物或缓存文件未上传或已被删除，属于环境/基础设施问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547826

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528548014

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528548045

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528548064

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528548150

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31724189348/job/94528547860) |


## [Run #31722157936](https://github.com/sgl-project/sglang/actions/runs/31722157936)
- **分支**: `cursor/fix-mooncake-local-hostname-20a4`
- **总耗时**: 212.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31722157936

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 67.7min | 精度回归 | Qwen3.5-9B GSM8K 精度测试失败，0/3 用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729974) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94560040633) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 健康检查快速失败，因同一PR中另一个作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94565839130) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，因同批次其他作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94568665661) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 test_npu_qwen3_5_9b_bf16_1p_gsm8k.py 运行约64分钟后退出码为1，所有3个用例均未通过，属于精度回归问题，可能由模型或推理逻辑改动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729974

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1128秒，未达到性能预期，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94560040633

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94565839130

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动阶段被PR健康检查拦截，检测到同批次base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，导致本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94568665661

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521518539) |
| multimodal-gen-test-1-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729211) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729329) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729384) |
| base-b-test-16-npu-a3 / run (0) | 80.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729400) |
| base-b-test-4-npu-a3 / run (1) | 10.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729422) |
| base-b-test-4-npu-a3 / run (0) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729494) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729542) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729545) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729743) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729745) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722157936/job/94521729830) |


## [Run #31722141924](https://github.com/sgl-project/sglang/actions/runs/31722141924)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 25.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31722141924

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521593929) |
| base-b-test-4-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594041) |
| base-b-test-8-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594056) |
| base-b-test-2-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594063) |
| base-b-test-16-npu-a3 / run (0) | 24.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594141) |
| base-b-test-4-npu-a3 / run (1) | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594181) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594229) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594360) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594373) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594474) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594498) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521593929

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理或路径错误，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594041

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594056

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594063

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594141

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于环境或资源配置问题，需检查相关存储路径或文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594181

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594229

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594360

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594373

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594474

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594498

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31722141924/job/94521594013) |


## [Run #31721938655](https://github.com/sgl-project/sglang/actions/runs/31721938655)
- **分支**: `refactor-mxfp4-sm100-trtllm-moerunner`
- **总耗时**: 47.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31721938655

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900169) |
| base-b-test-8-npu-a3 / run (0) | 46.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900382) |
| base-b-test-1-npu-a3 / run (0) | 46.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900392) |
| base-b-test-4-npu-a3 / run (1) | 46.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900408) |
| base-a-test-1-npu-a2 / run (0) | 4.8min | 环境问题 | NPU测试用例执行时出现未知应用异常，导致测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900421) |
| base-b-test-16-npu-a3 / run (0) | 46.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900425) |
| base-b-test-2-npu-a3 / run (0) | 46.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900463) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 46.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900641) |
| base-b-test-4-npu-a3 / run (0) | 46.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900643) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 46.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900705) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 46.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900766) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 46.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900910) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node.js弃用警告及上传artifact步骤，未显示multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900169

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900382

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置问题，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900392

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是缓存过期或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900408

- **base-a-test-1-npu-a2 / run (0)**: 测试文件test_npu_ascend_backend.py在运行21秒后报错ERR99999 UNKNOWN application exception，返回码1，最终作业退出码255。可能是NPU环境或依赖问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900421

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900425

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900463

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在 Azure Blob 存储中找到，可能是资源被清理或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900641

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900643

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900705

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900766

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个存储对象缺失或路径错误，可能是上传失败、过期或配置变更所致，需检查存储配置和文件存在性。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721938655/job/94520900910


## [Run #31721830435](https://github.com/sgl-project/sglang/actions/runs/31721830435)
- **分支**: `main`
- **总耗时**: 28.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31721830435

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545099) |
| base-b-test-1-npu-a3 / run (0) | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545280) |
| base-b-test-8-npu-a3 / run (0) | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545358) |
| base-b-test-16-npu-a3 / run (0) | 27.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545374) |
| base-b-test-2-npu-a3 / run (0) | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545395) |
| base-b-test-4-npu-a3 / run (1) | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545477) |
| base-b-test-4-npu-a3 / run (0) | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545534) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545800) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545816) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545843) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545984) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未包含任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545099

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545280

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545358

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545374

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545395

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545477

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545534

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545800

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545816

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545843

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545984

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721830435/job/94520545450) |


## [Run #31721323279](https://github.com/sgl-project/sglang/actions/runs/31721323279)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 9.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31721323279

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958086) |
| base-b-test-2-npu-a3 / run (0) | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958128) |
| base-b-test-4-npu-a3 / run (0) | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958194) |
| base-b-test-16-npu-a3 / run (0) | 9.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958199) |
| base-b-test-4-npu-a3 / run (1) | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958224) |
| base-b-test-8-npu-a3 / run (0) | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958293) |
| base-b-test-1-npu-a3 / run (0) | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958296) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958725) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958832) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958907) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518959073) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958086

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958128

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于环境或资源配置问题，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958194

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI依赖的某个存储对象缺失或路径错误，可能是上传失败、清理策略或配置变更所致，属于基础设施环境问题，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958199

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958224

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问一个不存在的存储对象（如模型权重、缓存或日志文件），需检查相关路径或上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958293

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958296

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958725

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958832

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958907

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518959073

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721323279/job/94518958161) |


## [Run #31721168304](https://github.com/sgl-project/sglang/actions/runs/31721168304)
- **分支**: `codex/extensible-serve-backends`
- **总耗时**: 240.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31721168304

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | NPU性能测试未通过，minimax_m2_5 w8a8 4p长序列测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94560005056) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94569513864) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94569998432) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94587456520) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试 test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py 执行1112秒后失败，0/1通过，属于性能测试未达标或执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94560005056

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，导致本作业在启动前被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94569513864

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（4-npu）作为级联失败被快速跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94569998432

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示根因失败作业为base-c-test-perf-8-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94587456520

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522888437) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522888489) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522888494) |
| base-b-test-4-npu-a3 / run (0) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522888642) |
| base-b-test-16-npu-a3 / run (0) | 51.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522888690) |
| base-b-test-4-npu-a3 / run (1) | 10.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522888712) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522888769) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522889768) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522889835) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522890089) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31721168304/job/94522890123) |


## [Run #31719494392](https://github.com/sgl-project/sglang/actions/runs/31719494392)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 21.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31719494392

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.1min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871331) |
| base-b-test-2-npu-a3 / run (0) | 20.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871521) |
| base-b-test-16-npu-a3 / run (0) | 20.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871573) |
| base-b-test-8-npu-a3 / run (0) | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871582) |
| base-b-test-4-npu-a3 / run (1) | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871618) |
| base-b-test-4-npu-a3 / run (0) | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871635) |
| base-b-test-1-npu-a3 / run (0) | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871683) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871961) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512872025) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512872086) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512872132) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871331

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象缺失，可能是构建产物未上传、路径错误或存储被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871521

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871573

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871582

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871618

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871635

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871683

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871961

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512872025

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512872086

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512872132

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31719494392/job/94512871639) |


## [Run #31719290961](https://github.com/sgl-project/sglang/actions/runs/31719290961)
- **分支**: `sushil/fid_benchmark`
- **总耗时**: 47.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31719290961

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 37.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31719290961/job/94512003292) |


## [Run #31719178903](https://github.com/sgl-project/sglang/actions/runs/31719178903)
- **分支**: `fix/aiter-diffusion-drop-unused-lse`
- **总耗时**: 42.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31719178903

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31719178903/job/94511623917) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31719178903/job/94511623917


## [Run #31718945455](https://github.com/sgl-project/sglang/actions/runs/31718945455)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 7.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31718945455

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 1.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887221) |
| base-b-test-8-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887700) |
| base-b-test-16-npu-a3 / run (0) | 5.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887708) |
| base-b-test-2-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887711) |
| base-b-test-4-npu-a3 / run (0) | 5.8min | AI调用失败 | HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887781) |
| base-b-test-1-npu-a3 / run (0) | 5.8min | AI调用失败 | HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887805) |
| base-b-test-4-npu-a3 / run (1) | 5.8min | AI调用失败 | HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887822) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.8min | AI调用失败 | HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510889008) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.8min | AI调用失败 | HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510889021) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510889025) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510889234) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887221

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887700

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887708

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887711

- **base-b-test-4-npu-a3 / run (0)**: HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out.
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887781

- **base-b-test-1-npu-a3 / run (0)**: HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out.
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887805

- **base-b-test-4-npu-a3 / run (1)**: HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out.
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887822

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out.
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510889008

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out.
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510889021

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510889025

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510889234

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718945455/job/94510887492) |


## [Run #31718403338](https://github.com/sgl-project/sglang/actions/runs/31718403338)
- **分支**: `lmzheng/model-extension-hooks`
- **总耗时**: 266.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31718403338

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.0min | 性能回归 | NPU性能测试未达预期，minimax_m2_5测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94555353214) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 健康检查发现同批次其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94563543197) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94578767159) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1151秒后退出码1，属于性能测试未通过，可能因推理速度或延迟未达50ms目标。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94555353214

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到同运行中的base-c-test-perf-8-npu-a3作业失败，将其判定为根因作业，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94563543197

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被跳过，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94578767159

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509187834) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509187860) |
| base-b-test-1-npu-a3 / run (0) | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509187975) |
| base-b-test-8-npu-a3 / run (0) | 9.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509187987) |
| base-b-test-16-npu-a3 / run (0) | 52.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188022) |
| base-b-test-4-npu-a3 / run (0) | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188041) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188072) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188269) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188562) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 45.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188589) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188650) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94509188749) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718403338/job/94562253092) |


## [Run #31718371605](https://github.com/sgl-project/sglang/actions/runs/31718371605)
- **分支**: `perf/dsv4-nonpaged-trivial-rows`
- **总耗时**: 236.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31718371605

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 26.8min | 性能回归 | NPU性能测试未通过，Qwen3-235B模型测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94553899482) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94570640600) |

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行1346秒后退出码为1，4个测试全部失败，疑似性能未达预期或运行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94553899482

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到另一个作业 base-c-test-perf-16-npu-a3 失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94570640600

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 43.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903117) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903229) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903370) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903374) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903378) |
| base-b-test-16-npu-a3 / run (0) | 63.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903380) |
| base-b-test-2-npu-a3 / run (0) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903389) |
| base-b-test-4-npu-a3 / run (0) | 32.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903467) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903625) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903662) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903801) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94508903985) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94547434865) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718371605/job/94554634662) |


## [Run #31718370243](https://github.com/sgl-project/sglang/actions/runs/31718370243)
- **分支**: `dsv4_state_pool_size`
- **总耗时**: 311.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31718370243

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.2min | 性能回归 | NPU性能测试未达标，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94545234780) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 52.2min | 性能回归 | qwen3_235b_a22b性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94550785390) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1059秒后失败，0/1通过，属于性能测试未达到预期指标，可能因模型推理速度或延迟不满足要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94545234780

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中qwen3_235b_a22b测试用例失败（exit code 1），而其他两个用例通过。该测试为50ms延迟目标，可能因模型性能未达预期或环境波动导致超时/不达标。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94550785390

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 47.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508895345) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508895661) |
| base-b-test-1-npu-a3 / run (0) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508895726) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508895757) |
| base-b-test-4-npu-a3 / run (1) | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508895758) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508895771) |
| base-b-test-4-npu-a3 / run (0) | 33.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508895873) |
| base-b-test-16-npu-a3 / run (0) | 51.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508895897) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508896291) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508896479) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508896561) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94508896570) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94552152290) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 80.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718370243/job/94572346120) |


## [Run #31718301030](https://github.com/sgl-project/sglang/actions/runs/31718301030)
- **分支**: `add_mxfp4w4a8_quantization_for_npu`
- **总耗时**: 267.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31718301030

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 38.1min | 性能回归 | NPU性能测试中kimi_k2_6用例失败，未达性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631789181) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.7min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631789247) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94640032764) |

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms用例退出码1，耗时1520秒，可能因性能不达标或运行错误导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631789181

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1082秒后退出码为1，属于性能测试未通过，可能因模型推理性能未达到预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631789247

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现base-c-test-perf-16-npu-a3和8-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于上游失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94640032764

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 34.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788400) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788636) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788683) |
| multimodal-gen-test-1-npu-a3 | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788686) |
| base-b-test-16-npu-a3 / run (0) | 55.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788776) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788917) |
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788927) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788965) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631788974) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631789006) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631789151) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 137.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718301030/job/94631789440) |


## [Run #31718123655](https://github.com/sgl-project/sglang/actions/runs/31718123655)
- **分支**: `swa-retain-to-mamba-checkpoint`
- **总耗时**: 295.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31718123655

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 1.8min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508107973) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.2min | 性能回归 | NPU性能测试未达预期，测试用例执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94541878000) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 47.2min | 性能回归 | NPU性能测试中kimi_k2_6用例失败，未达到性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94547658529) |

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 1.92 时，从内部缓存服务下载 channel-rust-1.92.toml 超时，导致脚本退出码非零，作业失败。属于基础设施网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508107973

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1059秒，未通过性能基准，可能因模型性能下降或环境波动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94541878000

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中deepseek_v4_flash和qwen3_5_397b通过，但kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms用例退出码1，耗时1552秒，可能因性能未达20ms目标或运行错误导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94547658529

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508107370) |
| base-b-test-1-npu-a3 / run (0) | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508107914) |
| base-b-test-4-npu-a3 / run (0) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508108086) |
| base-b-test-2-npu-a3 / run (0) | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508108150) |
| base-b-test-16-npu-a3 / run (0) | 46.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508108151) |
| base-b-test-4-npu-a3 / run (1) | 15.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508108284) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508108582) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508109162) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508109331) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508109347) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94508109368) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94550628731) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 81.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31718123655/job/94564366756) |


## [Run #31715508299](https://github.com/sgl-project/sglang/actions/runs/31715508299)
- **分支**: `minimax-m3-moe-dual-stream`
- **总耗时**: 463.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31715508299

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.0min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94537907469) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1053秒后失败，0/1通过，属于性能测试未达标，可能因模型推理速度或延迟不满足50ms目标导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94537907469

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499257858) |
| multimodal-gen-test-1-npu-a3 | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499257946) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499257977) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499257987) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499258020) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499258041) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499258057) |
| base-b-test-16-npu-a3 / run (0) | 52.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499258104) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499258473) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499258491) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499258528) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94499258682) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 272.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94546447068) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94546770254) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715508299/job/94563406220) |


## [Run #31715185419](https://github.com/sgl-project/sglang/actions/runs/31715185419)
- **分支**: `main`
- **总耗时**: 50.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31715185419

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31715185419/job/94498274897) |


## [Run #31714664459](https://github.com/sgl-project/sglang/actions/runs/31714664459)
- **分支**: `codex/component-residency-policy`
- **总耗时**: 256.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31714664459

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.7min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94537172238) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 上游作业失败导致快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94542959293) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94547413803) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业（8-npu perf）已失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94566537522) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1140秒后失败，退出码1，0/1通过，属于性能指标未达预期。
  链接: https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94537172238

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业因依赖的base-c-test-perf-8-npu-a3作业失败而被健康检查机制快速失败跳过，自身未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94542959293

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在健康检查阶段因检测到根因作业base-c-test-perf-8-npu-a3失败而触发fast-fail，未实际运行测试，属于级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94547413803

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的PR健康检查阶段被快速失败机制跳过，原因是同PR中base-c-test-perf-8-npu-a3作业已失败，属于级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94566537522

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497863940) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864185) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864202) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864321) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864330) |
| base-b-test-16-npu-a3 / run (0) | 56.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864355) |
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864380) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864479) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864800) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864897) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864983) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31714664459/job/94497864994) |


## [Run #31713837840](https://github.com/sgl-project/sglang/actions/runs/31713837840)
- **分支**: `feat/qwen35-shared-kv-verify`
- **总耗时**: 158.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31713837840

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103934) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495105143) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94523072015) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94534702675) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业base-c-test-perf-8-npu-a3，导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103934

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495105143

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1116秒，未通过性能基准要求，可能因模型推理延迟或吞吐量未达标导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94523072015

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94534702675

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103576) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103720) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103759) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103788) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103803) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103908) |
| base-b-test-16-npu-a3 / run (0) | 46.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495103959) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495104976) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 37.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495105194) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94495105216) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713837840/job/94530790253) |


## [Run #31713616678](https://github.com/sgl-project/sglang/actions/runs/31713616678)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 59.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31713616678

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 59.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781636) |
| base-b-test-4-npu-a3 / run (0) | 59.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781674) |
| base-b-test-8-npu-a3 / run (0) | 59.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781698) |
| base-b-test-16-npu-a3 / run (0) | 59.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781792) |
| base-b-test-4-npu-a3 / run (1) | 59.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781892) |
| base-b-test-1-npu-a3 / run (0) | 59.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781908) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 59.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492782005) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 59.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492782029) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 59.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492782143) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 59.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492782345) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781636

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781674

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781698

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781792

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781892

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物或依赖文件未正确上传，或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781908

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492782005

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492782029

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492782143

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492782345

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781521) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713616678/job/94492781627) |


## [Run #31713481119](https://github.com/sgl-project/sglang/actions/runs/31713481119)
- **分支**: `swa-retain-to-mamba-checkpoint`
- **总耗时**: 52.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31713481119

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 51.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438247) |
| base-b-test-4-npu-a3 / run (0) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438320) |
| base-b-test-2-npu-a3 / run (0) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438399) |
| base-b-test-8-npu-a3 / run (0) | 51.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438423) |
| base-b-test-1-npu-a3 / run (0) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438452) |
| base-b-test-4-npu-a3 / run (1) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438480) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438771) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438785) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438880) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492439057) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438247

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438320

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438399

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的工件/缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题，与代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438423

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438452

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438480

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438771

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438785

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438880

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失，可能是上传失败或路径错误，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492439057

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438038) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31713481119/job/94492438246) |


## [Run #31712200105](https://github.com/sgl-project/sglang/actions/runs/31712200105)
- **分支**: `codex/component-residency-policy`
- **总耗时**: 32.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31712200105

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917077) |
| base-b-test-2-npu-a3 / run (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917321) |
| base-b-test-1-npu-a3 / run (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917346) |
| base-b-test-16-npu-a3 / run (0) | 26.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917452) |
| base-b-test-8-npu-a3 / run (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917681) |
| base-b-test-4-npu-a3 / run (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917767) |
| base-b-test-4-npu-a3 / run (1) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917800) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487918197) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487918342) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487918348) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487918414) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败断言，仅有Node.js版本弃用警告和上传artifact时未找到文件（diffusion-failures/为空）的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917077

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是缓存或工件未正确上传，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917321

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917346

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是日志上传或依赖下载路径错误，属于基础设施配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917452

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917681

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917767

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917800

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487918197

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487918342

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487918348

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径错误，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487918414

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31712200105/job/94487917713) |


## [Run #31711897573](https://github.com/sgl-project/sglang/actions/runs/31711897573)
- **分支**: `marv/fuse_gdn_in_proj`
- **总耗时**: 215.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31711897573

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.8min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94519686848) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94527062979) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.0min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94528172564) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94546931549) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1089秒后退出码为1，属于性能测试未通过，可能因模型推理速度未达预期或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94519686848

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94527062979

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而触发fast-fail机制，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94528172564

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的PR健康检查中，检测到base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94546931549

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980007) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980041) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980061) |
| base-b-test-8-npu-a3 / run (0) | 8.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980068) |
| base-b-test-2-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980127) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980152) |
| base-b-test-16-npu-a3 / run (0) | 47.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980167) |
| base-b-test-4-npu-a3 / run (0) | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980183) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980667) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980712) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980765) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31711897573/job/94486980837) |


## [Run #31710585087](https://github.com/sgl-project/sglang/actions/runs/31710585087)
- **分支**: `bbuf/b300-ernie-qknorm-rope`
- **总耗时**: 53.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31710585087

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482334923) |
| base-b-test-1-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482334968) |
| base-b-test-2-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482334985) |
| base-b-test-8-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335080) |
| base-b-test-4-npu-a3 / run (1) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335133) |
| base-b-test-4-npu-a3 / run (0) | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335195) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335400) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335502) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335671) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 52.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335715) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482334923

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物或依赖文件未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482334968

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482334985

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施/环境配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335080

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是缓存或依赖文件未正确上传，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335133

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335195

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335400

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335502

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335671

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482335715

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482334891) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31710585087/job/94482334946) |


## [Run #31709621803](https://github.com/sgl-project/sglang/actions/runs/31709621803)
- **分支**: `mmangkad/fix-kimi-k3-deferred-backend-key`
- **总耗时**: 205.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31709621803

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94515189520) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因同PR中另一个作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94520204956) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94521217248) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94538534819) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1109秒后失败，该测试为性能测试，预计时间3600秒，但未通过性能指标要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94515189520

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于CI流程的级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94520204956

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，将其视为根因，导致本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94521217248

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因作业 base-c-test-perf-8-npu-a3 失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94538534819

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478978387) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478978576) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478978615) |
| base-b-test-16-npu-a3 / run (0) | 53.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478978646) |
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478978821) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478978848) |
| base-b-test-8-npu-a3 / run (0) | 10.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478978891) |
| base-b-test-4-npu-a3 / run (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478978900) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478979231) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478979282) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478979386) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709621803/job/94478979416) |


## [Run #31709151289](https://github.com/sgl-project/sglang/actions/runs/31709151289)
- **分支**: `marv/ar_norm_per_token_quant_fusion`
- **总耗时**: 246.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31709151289

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 54.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422740) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.8min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94507669242) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94519566524) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 作业因健康检查快速失败机制被跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94546129116) |

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因，导致本作业在启动前被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422740

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，退出码1，耗时1123秒，未达到性能预期，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94507669242

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94519566524

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示该作业在启动前被健康检查过滤，根因是另一个作业base-c-test-perf-8-npu-a3失败，导致本作业被级联跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94546129116

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477421853) |
| base-b-test-2-npu-a3 / run (0) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422098) |
| base-b-test-16-npu-a3 / run (0) | 45.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422156) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422265) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422269) |
| base-b-test-1-npu-a3 / run (0) | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422290) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422293) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422338) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422682) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422881) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 126.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31709151289/job/94477422933) |


## [Run #31707526148](https://github.com/sgl-project/sglang/actions/runs/31707526148)
- **分支**: `dsv4_fp8_trtllm_gen`
- **总耗时**: 220.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31707526148

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | 性能回归 | NPU性能测试未达预期，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94507581641) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 上游作业失败导致本作业被快速跳过，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94514956368) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94518567304) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业（perf-8-npu-a3）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94535904984) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时约1140秒，未通过性能测试，可能因性能未达标或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94507581641

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现同运行中的base-c-test-perf-8-npu-a3作业失败，被判定为根因，触发fast-fail机制，本作业未实际执行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94514956368

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，作为根因作业，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94518567304

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在健康检查阶段因根因作业 base-c-test-perf-8-npu-a3 失败而被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94535904984

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471841467) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471841528) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471841601) |
| base-b-test-2-npu-a3 / run (0) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471841605) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471841637) |
| base-b-test-16-npu-a3 / run (0) | 63.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471841654) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471841685) |
| base-b-test-4-npu-a3 / run (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471841702) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471842498) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471842500) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471842606) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707526148/job/94471842652) |


## [Run #31707437509](https://github.com/sgl-project/sglang/actions/runs/31707437509)
- **分支**: `sgl-router/upstream-lb-1-load-publisher`
- **总耗时**: 71.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31707437509

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387656) |
| base-b-test-2-npu-a3 / run (0) | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387662) |
| base-b-test-8-npu-a3 / run (0) | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387751) |
| base-b-test-4-npu-a3 / run (1) | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387827) |
| base-b-test-1-npu-a3 / run (0) | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387945) |
| base-b-test-16-npu-a3 / run (0) | 67.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388079) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388335) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388505) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388534) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 67.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388646) |

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/环境配置问题，需检查相关 blob 路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387656

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387662

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是上游作业未成功上传或存储配置问题，需检查相关依赖资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387751

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源（如模型权重、数据集或缓存）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387827

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387945

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388079

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388335

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388505

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388534

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472388646

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387466) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31707437509/job/94472387803) |


## [Run #31706940714](https://github.com/sgl-project/sglang/actions/runs/31706940714)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 211.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31706940714

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未达预期，minimax_m2_5测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94507151814) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因健康检查发现其他作业失败而快速失败，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94515639435) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94517607718) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.6min | 其他 | 健康检查快速失败，因其他作业（base-c-test-perf-8-npu-a3）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94529714759) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1096秒后退出码1，0/1通过，属于性能指标未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94507151814

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示该作业在启动阶段被健康检查拦截，原因是同一次运行中另一个作业（base-c-test-perf-8-npu-a3）已失败，触发了fast-fail机制，导致本作业未实际执行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94515639435

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动阶段被PR健康检查拦截，原因是同一次运行中base-c-test-perf-8-npu-a3作业已失败，本作业作为级联失败被快速跳过，并非自身执行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94517607718

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在健康检查阶段检测到根因作业base-c-test-perf-8-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94529714759

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469868943) |
| base-b-test-2-npu-a3 / run (0) | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469869063) |
| base-b-test-1-npu-a3 / run (0) | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469869108) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469869144) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469869229) |
| base-b-test-4-npu-a3 / run (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469869302) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469869339) |
| base-b-test-16-npu-a3 / run (0) | 63.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469869376) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469869928) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469870033) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469870037) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31706940714/job/94469870170) |


---
*Auto-generated by npu_pr_monitor.py*