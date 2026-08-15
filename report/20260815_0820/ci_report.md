# NPU CI 执行监控
**生成时间**: 2026-08-15 00:20 UTC
**分析 Run 数**: 30

---

## 📊 本次执行总结

- **成功 Job 数**: 217
- **失败 Run 数**: 27
- **成功 Job 平均耗时**: 30.0min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 277.4min | #31792405759 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94857085741) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 140.7min | #31800949997 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768623012) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 127.6min | #31800473220 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767102354) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 112.8min | #31792405759 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742114207) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.7min | #31811018271 | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500998) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.5min | #31805859923 | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784658179) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.5min | #31792190001 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741461114) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.1min | #31792304321 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803743) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.1min | #31801506478 | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774108024) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.8min | #31792569595 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742633366) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31792405759<br>[#29723 [AMD] Add fused all-reduce RMSNorm per-token FP8/MXFP4 quant](https://github.com/sgl-project/sglang/pull/29723) | `marv/ar_norm_per_token_quant_fusion` | 802.1min | 1 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31792405759) |
| #31792304321<br>[#28655 [AMD] GDN linear out-proj fusion](https://github.com/sgl-project/sglang/pull/28655) | `marv/gdn_out_proj_fusion` | 640.2min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31792304321) |
| #31792569595<br>[#33068 [AMD] Fuse quantized in_proj layers in Qwen3.5](https://github.com/sgl-project/sglang/pull/33068) | `marv/fuse_gdn_in_proj` | 634.5min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31792569595) |
| #31800949997<br>[#34277 [DSV4] Emit TMA-aligned UE8M0 scales for FP8 einsum](https://github.com/sgl-project/sglang/pull/34277) | `dsv4/pack-tma-for-einsum` | 575.0min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31800949997) |
| #31800473220<br>[#34565 [Unified Tree] Support Branching-Point Caching for the SWA Component](https://github.com/sgl-project/sglang/pull/34565) | `cjc/unified-swa-branching` | 553.9min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31800473220) |
| #31801506478<br>[#34844 [Spec] Support MegaMoE for DSpark under dp attention](https://github.com/sgl-project/sglang/pull/34844) | `dspark-megamoe-ep` | 550.8min | 5 | base-b-test-16-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31801506478) |
| #31792190001<br>[#34781 fix(muse-glimmer): parse required/named tool calls natively](https://github.com/sgl-project/sglang/pull/34781) | `muse-glimmer-required-native-toolcall` | 542.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31792190001) |
| #31798419611<br>[#33005 [FP8][MoE] Honor UE8M0 activation scales in Triton MoE](https://github.com/sgl-project/sglang/pull/33005) | `dsv4-triton-moe-ue8m0-scales` | 523.1min | 4 | base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31798419611) |
| #31798534086<br>[#32963 [NVIDIA][comm] Merge EP+MoE-TP post-experts all-reduces into one _TP reduction](https://github.com/sgl-project/sglang/pull/32963) | `fix-hybrid-ep-tp-allreduce-fusion` | 522.9min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31798534086) |
| #31786806619<br>[#27010 [HiCache] Fix PP inconsistency with HiCache L3 (#22607)](https://github.com/sgl-project/sglang/pull/27010) | `sglang_pp_bug4` | 521.4min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31786806619) |
| #31805859923<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 509.3min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31805859923) |
| #31787949551<br>[#32313 [Feature] Optimize TP LMHead with All-to-All](https://github.com/sgl-project/sglang/pull/32313) | `lm-head-opt` | 507.4min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31787949551) |
| #31798317506<br>[#34561 [Fix] Fix Nemotron-H Mamba illegal memory access under DP attention with CUDA graph](https://github.com/sgl-project/sglang/pull/34561) | `nemotron-dp-acc` | 504.4min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31798317506) |
| #31811018271<br>[#32340 Amd/dsv4 shared experts fusion top6](https://github.com/sgl-project/sglang/pull/32340) | `amd/dsv4-shared-experts-fusion-top6` | 468.7min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31811018271) |
| #31791333305<br>[#34299 [KDA] Add zero-copy native prefill checkpoints and packed decode](https://github.com/sgl-project/sglang/pull/34299) | `codex/sglang-phase-a-admission-rebased-20260810` | 457.4min | 4 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31791333305) |
| #31810612819<br>[#34693 [Kernel] Replace dsv3_router_gemm with the unified tiny GEMM](https://github.com/sgl-project/sglang/pull/34693) | `tiny-gemm-unify-router` | 449.9min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31810612819) |
| #31799471057<br>[#32114 Delete cutlass_mla, non-Marlin GPTQ, AWQ AOT kernel, and Dual Chunk Flash Attention](https://github.com/sgl-project/sglang/pull/32114) | `delete-cutlass-mla-gptq-awq-dualchunk` | 322.9min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31799471057) |
| #31799470265<br>[#33930 Clean logging under --weight-loader-prefetch-checkpoints](https://github.com/sgl-project/sglang/pull/33930) | `brayden/clean-startup-logs` | 321.8min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31799470265) |
| #31787169329<br>[#34823 Skip oow slot freeing under eagle](https://github.com/sgl-project/sglang/pull/34823) | `main` | 168.8min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31787169329) |
| #31795236776<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 144.9min | 10 | base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31795236776) |
| #31789548064<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 84.5min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31789548064) |
| #31808303515<br>[#32944 [MoE] Fuse swiglu moe up gemm epilogue](https://github.com/sgl-project/sglang/pull/32944) | `main` | 69.0min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31808303515) |
| #31785237525<br>[#34242 [diffusion] Warn when BCG disables Cache-DiT](https://github.com/sgl-project/sglang/pull/34242) | `main` | 28.1min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31785237525) |
| #31813855774<br>[#30531 [DSA] Skip indexer KV cache for skip-topk layers](https://github.com/sgl-project/sglang/pull/30531) | `mmangkad/reland-30310` | 25.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31813855774) |
| #31788566695<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 14.5min | 12 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31788566695) |
| #31806497942<br>[#34304 Remove the torchao integration (--torchao-config)](https://github.com/sgl-project/sglang/pull/34304) | `main` | 12.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31806497942) |
| #31807515154<br>[#34542 [MiniMax-M3] Overlap shared and routed experts](https://github.com/sgl-project/sglang/pull/34542) | `main` | 6.3min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31807515154) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 26 | 24 | 0 | 2 |
| base-b-test-1-npu-a3 / run (0) | 26 | 16 | 0 | 10 |
| base-b-test-16-npu-a3 / run (0) | 26 | 15 | 1 | 10 |
| base-b-test-2-npu-a3 / run (0) | 26 | 16 | 0 | 10 |
| base-b-test-4-npu-a3 / run (0) | 26 | 16 | 0 | 10 |
| base-b-test-4-npu-a3 / run (1) | 26 | 16 | 0 | 10 |
| base-b-test-8-npu-a3 / run (0) | 26 | 16 | 0 | 10 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26 | 15 | 1 | 10 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26 | 15 | 0 | 11 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26 | 16 | 0 | 10 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 26 | 16 | 0 | 10 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 15 | 1 | 0 | 14 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 15 | 3 | 0 | 12 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 16 | 8 | 0 | 8 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16 | 0 | 0 | 16 |
| multimodal-gen-test-1-npu-a3 | 29 | 24 | 0 | 5 |

---


## [Run #31814050432](https://github.com/sgl-project/sglang/actions/runs/31814050432)
- **分支**: `agent/fix-diffusion-weight-lock-filename`
- **总耗时**: 77.8min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31814050432

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31814050432/job/94811431653) |


## [Run #31813855774](https://github.com/sgl-project/sglang/actions/runs/31813855774)
- **分支**: `mmangkad/reland-30310`
- **总耗时**: 25.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31813855774

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810815966) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816190) |
| base-b-test-16-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816345) |
| base-b-test-2-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816384) |
| base-b-test-8-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816417) |
| base-b-test-4-npu-a3 / run (0) | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816511) |
| base-b-test-4-npu-a3 / run (1) | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816543) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810817285) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810817415) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810817440) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810817581) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810815966

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816190

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816345

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816384

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816417

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816511

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816543

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810817285

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810817415

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810817440

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810817581

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31813855774/job/94810816463) |


## [Run #31811018271](https://github.com/sgl-project/sglang/actions/runs/31811018271)
- **分支**: `amd/dsv4-shared-experts-fusion-top6`
- **总耗时**: 468.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31811018271

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.7min | 性能回归 | NPU性能测试未通过，minimax_m2_5 w8a8 4p测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94892357411) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因同批次其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94900451179) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94911065047) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1091秒后失败，0/1测试通过，属于性能测试未达标或执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94892357411

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94900451179

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而触发fast-fail，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94911065047

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801499722) |
| base-b-test-1-npu-a3 / run (0) | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500005) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500021) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500095) |
| base-b-test-4-npu-a3 / run (0) | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500229) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500350) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500362) |
| base-b-test-16-npu-a3 / run (0) | 46.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500377) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500881) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 109.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801500998) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801501104) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94801501141) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31811018271/job/94895581670) |


## [Run #31810612819](https://github.com/sgl-project/sglang/actions/runs/31810612819)
- **分支**: `tiny-gemm-unify-router`
- **总耗时**: 449.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31810612819

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.6min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94890442792) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 43.6min | 性能回归 | NPU性能测试中qwen3_235b用例失败，疑似性能未达标 | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94891455188) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94901496699) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1174秒后退出码为1，0/1通过，属于性能指标未满足预期，可能因模型推理速度或延迟未达基准。
  链接: https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94890442792

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中qwen3_235b_w8a8_8p_in3k5_out1k5_50ms用例退出码1，而其他两个用例通过。该用例名称含'50ms'，表明有性能指标要求，失败可能因实际性能未达到预期阈值，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94891455188

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3和16-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94901496699

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800180232) |
| multimodal-gen-test-1-npu-a3 | 36.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800180297) |
| base-b-test-16-npu-a3 / run (0) | 71.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800180328) |
| base-b-test-8-npu-a3 / run (0) | 10.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800180337) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800180430) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800180472) |
| base-b-test-4-npu-a3 / run (1) | 14.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800180479) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800180511) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800181548) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800181612) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800181653) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94800181774) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31810612819/job/94892617322) |


## [Run #31808303515](https://github.com/sgl-project/sglang/actions/runs/31808303515)
- **分支**: `main`
- **总耗时**: 69.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31808303515

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 68.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582118) |
| base-b-test-4-npu-a3 / run (1) | 68.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582222) |
| base-b-test-8-npu-a3 / run (0) | 68.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582238) |
| base-b-test-1-npu-a3 / run (0) | 68.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582294) |
| base-b-test-4-npu-a3 / run (0) | 68.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582320) |
| base-b-test-16-npu-a3 / run (0) | 68.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582322) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 68.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582532) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 68.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582542) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 68.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582642) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 68.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582666) |

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储资源缺失，可能是构建产物或依赖文件未正确上传，或路径配置错误。建议检查CI流程中相关存储访问配置及文件上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582118

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源（如模型权重或缓存文件）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582222

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582238

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是缓存、模型权重或日志文件未正确上传或已被删除，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582294

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582320

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582322

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582532

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582542

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582642

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582666

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792581997) |
| base-a-test-1-npu-a2 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31808303515/job/94792582184) |


## [Run #31807515154](https://github.com/sgl-project/sglang/actions/runs/31807515154)
- **分支**: `main`
- **总耗时**: 6.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31807515154

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005590) |
| base-b-test-4-npu-a3 / run (1) | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005803) |
| base-b-test-4-npu-a3 / run (0) | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005862) |
| base-b-test-2-npu-a3 / run (0) | 5.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005878) |
| base-b-test-1-npu-a3 / run (0) | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005900) |
| base-b-test-16-npu-a3 / run (0) | 5.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005962) |
| base-b-test-8-npu-a3 / run (0) | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006018) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | 其他 | 作业实际成功，但被标记为失败，可能是基础设施或状态同步问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006020) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006377) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006520) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006539) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006564) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005590

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005803

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005862

- **base-b-test-2-npu-a3 / run (0)**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于环境或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005878

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005900

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790005962

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是构建产物或依赖未正确上传，需检查存储配置或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006018

- **base-a-test-1-npu-a2 / run (0)**: 日志显示所有测试通过（2/2 passed），无错误或超时。失败可能源于GitHub Actions runner的Node 20弃用警告或作业状态误报，属于基础设施或平台问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006020

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006377

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006520

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006539

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31807515154/job/94790006564


## [Run #31806497942](https://github.com/sgl-project/sglang/actions/runs/31806497942)
- **分支**: `main`
- **总耗时**: 12.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31806497942

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670397) |
| base-b-test-1-npu-a3 / run (0) | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670536) |
| base-b-test-8-npu-a3 / run (0) | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670546) |
| base-b-test-2-npu-a3 / run (0) | 11.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670583) |
| base-b-test-4-npu-a3 / run (0) | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670625) |
| base-b-test-16-npu-a3 / run (0) | 11.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670641) |
| base-b-test-4-npu-a3 / run (1) | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670656) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786671059) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786671078) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786671133) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786671136) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670397

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是缓存或依赖文件未正确上传，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670536

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670546

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670583

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/存储配置问题，需检查上传步骤或清理过期缓存。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670625

- **base-b-test-16-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Blob 文件（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670641

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670656

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786671059

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到。可能是 CI 配置中引用的 blob 路径错误、资源被删除或尚未上传，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786671078

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，属于外部存储环境问题，需检查资源是否存在或更新路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786671133

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786671136

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31806497942/job/94786670648) |


## [Run #31805859923](https://github.com/sgl-project/sglang/actions/runs/31805859923)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 509.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31805859923

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.2min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94881796484) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94888760854) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | PR健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94889121631) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94905469476) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1127秒后失败，0/1通过，属于性能测试未达到预期标准。
  链接: https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94881796484

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，根因失败被过滤后触发fast-fail，本作业未实际运行即被终止，属于上游作业失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94888760854

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查显示base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而触发fast-fail，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94889121631

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在健康检查阶段因根因作业base-c-test-perf-8-npu-a3失败而触发fast-fail，本作业未实际运行测试，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94905469476

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784657639) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784657765) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784657784) |
| base-b-test-16-npu-a3 / run (0) | 61.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784657808) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784657868) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784657927) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784657962) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784657992) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784658163) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784658179) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784658208) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31805859923/job/94784658303) |


## [Run #31801506478](https://github.com/sgl-project/sglang/actions/runs/31801506478)
- **分支**: `dspark-megamoe-ep`
- **总耗时**: 550.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31801506478

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 25.3min | 环境问题 | NPU Pod 启动失败，状态为 Failed，导致作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107571) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.3min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94876204270) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94884709489) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94886579108) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94901886211) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 Kubernetes Pod (linux-aarch64-a3-16-cn12-001-772vk-runner-6w2z9-workflow) 在启动后处于不健康状态，phase 为 Failed，可能是镜像拉取失败、资源不足或初始化错误，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107571

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1071秒后失败，该测试为性能测试，预期时间3600秒，实际未通过，可能因性能未达阈值或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94876204270

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行PR健康检查时，检测到同一运行中其他两个根因作业（base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3）已失败，触发fast-fail逻辑，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94884709489

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到根因作业 base-b-test-16-npu-a3/run 和 base-c-test-perf-8-npu-a3 失败，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94886579108

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94901886211

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107407) |
| base-b-test-4-npu-a3 / run (0) | 27.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107483) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107495) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107505) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107542) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107640) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107717) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107965) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 42.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774107986) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774108013) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31801506478/job/94774108024) |


## [Run #31800949997](https://github.com/sgl-project/sglang/actions/runs/31800949997)
- **分支**: `dsv4/pack-tma-for-einsum`
- **总耗时**: 575.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31800949997

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94875020088) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94882707946) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94885752099) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94904743383) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，运行时间1163秒，未通过性能基准要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94875020088

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，根据快速失败策略，本作业被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94882707946

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业在健康检查阶段因根因作业base-c-test-perf-8-npu-a3失败而被快速失败跳过，并非本作业自身问题，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94885752099

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，将其视为根因，本作业作为级联失败被跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94904743383

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768622668) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768622743) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768622747) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768622762) |
| base-b-test-16-npu-a3 / run (0) | 84.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768622785) |
| base-b-test-2-npu-a3 / run (0) | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768622817) |
| base-b-test-4-npu-a3 / run (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768622821) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 140.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768623012) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768623028) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 50.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768623033) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800949997/job/94768623135) |


## [Run #31800473220](https://github.com/sgl-project/sglang/actions/runs/31800473220)
- **分支**: `cjc/unified-swa-branching`
- **总耗时**: 553.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31800473220

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.6min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94865462119) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94879756555) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.5min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94895220863) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行约1010秒后退出码为1，0/1通过，属于性能指标未达标导致的回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94865462119

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94879756555

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，实际未执行测试，属于上游失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94895220863

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767101875) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767101890) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767101924) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767101949) |
| base-b-test-16-npu-a3 / run (0) | 46.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767101953) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767101975) |
| base-b-test-2-npu-a3 / run (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767102033) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767102073) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 127.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767102354) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767102360) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767102383) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 55.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94767102572) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800473220/job/94875002499) |


## [Run #31800314330](https://github.com/sgl-project/sglang/actions/runs/31800314330)
- **分支**: `mmangkad/fix-minimax-h3-bcg-cache-dit`
- **总耗时**: 50.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31800314330

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31800314330/job/94768796782) |


## [Run #31799471057](https://github.com/sgl-project/sglang/actions/runs/31799471057)
- **分支**: `delete-cutlass-mla-gptq-awq-dualchunk`
- **总耗时**: 322.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31799471057

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 322.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866464) |
| base-b-test-16-npu-a3 / run (0) | 322.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866499) |
| base-b-test-4-npu-a3 / run (0) | 322.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866517) |
| base-b-test-4-npu-a3 / run (1) | 322.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866587) |
| base-b-test-8-npu-a3 / run (0) | 322.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866641) |
| base-b-test-2-npu-a3 / run (0) | 322.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866710) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 322.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866928) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 322.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866938) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 322.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866964) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 322.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763867053) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866464

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866499

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866517

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失的环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866587

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866641

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866710

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866928

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866938

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866964

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763867053

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866393) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31799471057/job/94763866572) |


## [Run #31799470265](https://github.com/sgl-project/sglang/actions/runs/31799470265)
- **分支**: `brayden/clean-startup-logs`
- **总耗时**: 321.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31799470265

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763864995) |
| base-b-test-16-npu-a3 / run (0) | 321.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865003) |
| base-b-test-2-npu-a3 / run (0) | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865021) |
| base-b-test-4-npu-a3 / run (1) | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865047) |
| base-b-test-8-npu-a3 / run (0) | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865053) |
| base-b-test-4-npu-a3 / run (0) | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865077) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865244) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865248) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865266) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 321.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865343) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置不一致，需检查相关存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763864995

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象缺失，可能是日志上传失败、路径错误或资源被清理，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865003

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865021

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865047

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败、过期或配置错误，需检查相关存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865053

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865077

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865244

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865248

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865266

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865343

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763864857) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31799470265/job/94763865056) |


## [Run #31798534086](https://github.com/sgl-project/sglang/actions/runs/31798534086)
- **分支**: `fix-hybrid-ep-tp-allreduce-fusion`
- **总耗时**: 522.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31798534086

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.3min | 性能回归 | NPU性能测试未达预期，minimax_m2_5测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94864033689) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 41.2min | 性能回归 | NPU性能测试中qwen3_235b_a22b用例失败，疑似性能未达标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94873919973) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败机制触发，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94884962117) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1062秒后失败，该测试为性能测试，预计耗时3600秒，但提前退出且未通过，表明性能指标未达标。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94864033689

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中两个用例通过，但test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py退出码为1，耗时1283秒，远超其他用例，可能因性能不满足50ms延迟要求导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94873919973

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3和base-c-test-perf-16-npu-a3两个根因失败作业，触发fast-fail机制，本作业未实际运行即被终止，属于CI流程控制而非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94884962117

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760953772) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760953834) |
| base-b-test-8-npu-a3 / run (0) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760953864) |
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760953879) |
| base-b-test-4-npu-a3 / run (0) | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760953909) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760953939) |
| base-b-test-16-npu-a3 / run (0) | 57.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760954012) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760954015) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760954238) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760954284) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760954309) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94760954354) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798534086/job/94873339665) |


## [Run #31798419611](https://github.com/sgl-project/sglang/actions/runs/31798419611)
- **分支**: `dsv4-triton-moe-ue8m0-scales`
- **总耗时**: 523.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31798419611

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.5min | 环境问题 | Kubernetes Pod 启动失败，状态为 Failed，导致作业无法运行。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611672) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94860752818) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94871695352) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94882725309) |

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 Pod linux-aarch64-a3-16-cn12-001-772vk-runner-tx5xk-workflow 不健康且状态为 Failed，可能是资源分配、镜像拉取或节点问题，属于基础设施环境故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611672

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示健康检查检测到根因失败作业 base-c-test-acc-16-npu-a3，触发 fast-fail 跳过当前 perf 作业，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94860752818

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业base-c-test-acc-16-npu-a3，本作业被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94871695352

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因作业base-c-test-acc-16-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94882725309

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611239) |
| base-b-test-16-npu-a3 / run (0) | 54.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611344) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611393) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611443) |
| base-b-test-4-npu-a3 / run (1) | 16.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611444) |
| base-b-test-8-npu-a3 / run (0) | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611483) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611491) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611520) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611691) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611723) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798419611/job/94760611766) |


## [Run #31798317506](https://github.com/sgl-project/sglang/actions/runs/31798317506)
- **分支**: `nemotron-dp-acc`
- **总耗时**: 504.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31798317506

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 性能回归 | NPU性能测试未达到预期指标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94856246356) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 上游作业失败导致本作业被快速失败跳过，非本作业自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94863608062) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查检测到其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94866857228) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业（8卡性能测试）已失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94881168600) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行约1104秒后退出码为1，0/1通过，属于性能测试未达标（可能吞吐或延迟不满足要求），非环境或代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94856246356

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现同PR中base-c-test-perf-8-npu-a3作业失败，被判定为根因作业，触发fast-fail机制，本作业未实际运行即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94863608062

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动前的健康检查中发现base-c-test-perf-8-npu-a3作业失败，被判定为根因作业，本作业作为级联失败被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94866857228

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的PR健康检查中发现根因作业base-c-test-perf-8-npu-a3失败，触发fast-fail机制，本作业被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94881168600

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760317826) |
| base-b-test-2-npu-a3 / run (0) | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760317989) |
| base-b-test-16-npu-a3 / run (0) | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318042) |
| base-b-test-4-npu-a3 / run (1) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318104) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318143) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318144) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318177) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318194) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318889) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318893) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318904) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31798317506/job/94760318927) |


## [Run #31795236776](https://github.com/sgl-project/sglang/actions/runs/31795236776)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 144.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31795236776

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865234) |
| base-b-test-4-npu-a3 / run (0) | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865290) |
| base-b-test-1-npu-a3 / run (0) | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865306) |
| base-b-test-16-npu-a3 / run (0) | 144.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865325) |
| base-b-test-4-npu-a3 / run (1) | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865330) |
| base-b-test-2-npu-a3 / run (0) | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865396) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865561) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865588) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865664) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 144.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865710) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865234

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865290

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865306

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的存储对象缺失，可能是构建产物未上传、路径错误或存储被清理，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865325

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865330

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865396

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865561

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865588

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865664

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865710

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865189) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31795236776/job/94750865309) |


## [Run #31792569595](https://github.com/sgl-project/sglang/actions/runs/31792569595)
- **分支**: `marv/fuse_gdn_in_proj`
- **总耗时**: 634.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31792569595

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.1min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8测试用例失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94849856502) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 25.1min | 性能回归 | NPU性能测试未通过，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94860505725) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1121秒后失败，0/1测试通过，属于性能测试未达标或执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94849856502

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中qwen3_235b_a22b模型的w8a8_8p_in3k5_out1k5_50ms测试用例返回退出码1，测试耗时1277秒，未达到预期性能标准，导致整体测试0/4通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94860505725

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 30.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742632933) |
| base-b-test-16-npu-a3 / run (0) | 47.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742632956) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742632976) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742632978) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742632987) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742633003) |
| multimodal-gen-test-1-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742633004) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742633014) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742633286) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742633360) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 90.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742633366) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94742633589) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94858613854) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792569595/job/94868932686) |


## [Run #31792405759](https://github.com/sgl-project/sglang/actions/runs/31792405759)
- **分支**: `marv/ar_norm_per_token_quant_fusion`
- **总耗时**: 802.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31792405759

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94848629284) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1125秒后退出码1，属于性能测试未通过，可能因吞吐或延迟未达预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94848629284

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742113718) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742113854) |
| base-b-test-2-npu-a3 / run (0) | 19.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742113866) |
| base-b-test-16-npu-a3 / run (0) | 56.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742113868) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742113877) |
| base-b-test-4-npu-a3 / run (1) | 16.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742113886) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742113890) |
| base-b-test-1-npu-a3 / run (0) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742113912) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742114157) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742114172) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 112.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742114207) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94742114262) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94856013720) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 277.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94857085741) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 77.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792405759/job/94871688989) |


## [Run #31792304321](https://github.com/sgl-project/sglang/actions/runs/31792304321)
- **分支**: `marv/gdn_out_proj_fusion`
- **总耗时**: 640.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31792304321

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | NPU性能测试用例失败，未达到预期性能指标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94842981252) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 25.0min | 性能回归 | NPU性能测试未通过，qwen3_235b模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94854832423) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，耗时1117秒，未通过性能测试，可能因模型性能未达标或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94842981252

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行1268秒后失败，4个测试全部未通过，可能因性能未达预期或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94854832423

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803239) |
| base-b-test-4-npu-a3 / run (0) | 27.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803330) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803392) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803411) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803418) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803484) |
| base-b-test-16-npu-a3 / run (0) | 53.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803506) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803598) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803743) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803768) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803800) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94741803816) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94855327241) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792304321/job/94870065670) |


## [Run #31792190001](https://github.com/sgl-project/sglang/actions/runs/31792190001)
- **分支**: `muse-glimmer-required-native-toolcall`
- **总耗时**: 542.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31792190001

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试未达标，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94841772458) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94852241242) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94852911899) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94862617971) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1115秒后失败，退出码1，0/1测试通过。该测试为性能测试，可能因性能未达到预期阈值（如50ms延迟要求）而失败，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94841772458

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，根据快速失败策略，本作业被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94852241242

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94852911899

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 健康检查显示根因失败作业为base-c-test-perf-8-npu-a3，本作业（2-npu）被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94862617971

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741460443) |
| base-b-test-2-npu-a3 / run (0) | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741460561) |
| base-a-test-1-npu-a2 / run (0) | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741460589) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741460619) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741460710) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741460735) |
| base-b-test-16-npu-a3 / run (0) | 56.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741460792) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741460822) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741461014) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741461075) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741461114) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31792190001/job/94741461215) |


## [Run #31791932447](https://github.com/sgl-project/sglang/actions/runs/31791932447)
- **分支**: `agent/minimax-h3-b300-high-quality`
- **总耗时**: 51.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31791932447

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791932447/job/94742486453) |


## [Run #31791333305](https://github.com/sgl-project/sglang/actions/runs/31791333305)
- **分支**: `codex/sglang-phase-a-admission-rebased-20260810`
- **总耗时**: 457.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31791333305

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 78.2min | 精度回归 | NPU精度测试失败，qwen3_5_9b_bf16_1p_gsm8k测试用例返回退出码1，0/3通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738780169) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94831880154) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94839743276) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94842960705) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试套件base-c-test-acc-2-npu-a3中，qwen3_5_9b_bf16_1p_gsm8k精度测试失败，耗时4444秒超过预估3600秒，最终0/3用例通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738780169

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1114秒后失败，该测试为性能测试，要求50ms延迟，可能因性能未达标或环境问题导致退出码1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94831880154

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，触发 fast-fail 机制，本作业未实际运行即被取消，属于上游失败导致的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94839743276

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-acc-2-npu-a3和base-c-test-perf-8-npu-a3两个根因失败作业，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94842960705

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738779562) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738779615) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738779678) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738779703) |
| base-b-test-16-npu-a3 / run (0) | 57.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738779726) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738779739) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738779799) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738779842) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738780133) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 36.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738780220) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31791333305/job/94738780249) |


## [Run #31789548064](https://github.com/sgl-project/sglang/actions/runs/31789548064)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 84.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31789548064

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 83.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280764) |
| base-b-test-1-npu-a3 / run (0) | 83.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280822) |
| base-b-test-8-npu-a3 / run (0) | 83.7min | 环境问题 | 日志下载失败，Blob不存在 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280860) |
| base-b-test-16-npu-a3 / run (0) | 83.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280887) |
| base-b-test-4-npu-a3 / run (1) | 83.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280906) |
| base-b-test-4-npu-a3 / run (0) | 83.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280975) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733281623) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 83.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733281665) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 83.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733281691) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 83.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733281714) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280764

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于环境配置或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280822

- **base-b-test-8-npu-a3 / run (0)**: 作业日志显示Azure Blob存储返回BlobNotFound错误，说明日志文件已被删除或路径错误，无法获取实际运行信息，属于基础设施/存储问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280860

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280887

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280906

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280975

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733281623

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733281665

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储账户中找到，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733281691

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733281714

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280688) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31789548064/job/94733280909) |


## [Run #31788566695](https://github.com/sgl-project/sglang/actions/runs/31788566695)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 14.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31788566695

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730066847) |
| base-a-test-1-npu-a2 / run (0) | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730066975) |
| base-b-test-4-npu-a3 / run (1) | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730066983) |
| base-b-test-8-npu-a3 / run (0) | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730066993) |
| base-b-test-4-npu-a3 / run (0) | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067005) |
| base-b-test-2-npu-a3 / run (0) | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067008) |
| base-b-test-16-npu-a3 / run (0) | 13.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067043) |
| base-b-test-1-npu-a3 / run (0) | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067060) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067188) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067307) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067329) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067401) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤，无法判断失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730066847

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730066975

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730066983

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730066993

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，请求的资源在存储中不存在，可能是文件被删除、路径错误或上传未完成，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067005

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或路径错误，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067008

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067043

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067060

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067188

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067307

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067329

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31788566695/job/94730067401


## [Run #31787949551](https://github.com/sgl-project/sglang/actions/runs/31787949551)
- **分支**: `lm-head-opt`
- **总耗时**: 507.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31787949551

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.7min | 性能回归 | NPU性能测试未达预期，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94826566440) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.2min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94837352939) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94839192078) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 2.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94845112957) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时1155秒，未通过性能基准要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94826566440

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（16-npu）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94837352939

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 本作业在启动阶段被PR健康检查拦截，检测到根因作业base-c-test-perf-8-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94839192078

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因作业 base-c-test-perf-8-npu-a3 失败，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94845112957

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728117876) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728117988) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118137) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118140) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118143) |
| base-b-test-8-npu-a3 / run (0) | 8.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118170) |
| base-b-test-16-npu-a3 / run (0) | 52.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118176) |
| base-b-test-4-npu-a3 / run (1) | 15.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118189) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118449) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118480) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118493) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787949551/job/94728118584) |


## [Run #31787169329](https://github.com/sgl-project/sglang/actions/runs/31787169329)
- **分支**: `main`
- **总耗时**: 168.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31787169329

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724466) |
| base-b-test-4-npu-a3 / run (0) | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724546) |
| base-b-test-4-npu-a3 / run (1) | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724547) |
| base-b-test-16-npu-a3 / run (0) | 168.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724566) |
| base-b-test-8-npu-a3 / run (0) | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724574) |
| base-b-test-1-npu-a3 / run (0) | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724603) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724893) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724942) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724966) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 168.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724984) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查 blob 路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724466

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724546

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724547

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724566

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724574

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724603

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724893

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724942

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724966

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724984

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724482) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31787169329/job/94725724511) |


## [Run #31786806619](https://github.com/sgl-project/sglang/actions/runs/31786806619)
- **分支**: `sglang_pp_bug4`
- **总耗时**: 521.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31786806619

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.1min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94824403388) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94833838577) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94842900682) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行约19分钟后退出码为1，属于性能测试未通过，可能因吞吐或延迟未达预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94824403388

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到同批次作业 base-c-test-perf-8-npu-a3 失败，被判定为根因失败，因此本作业（base-c-test-perf-16-npu-a3）被快速失败机制跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94833838577

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业在启动前的PR健康检查阶段被快速失败机制跳过，原因是同一次运行中base-c-test-perf-8-npu-a3作业已失败，属于级联跳过，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94842900682

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724525765) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724525886) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724525896) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724525942) |
| base-a-test-1-npu-a2 / run (0) | 10.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724525973) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724526053) |
| base-b-test-16-npu-a3 / run (0) | 46.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724526062) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724526101) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724526265) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724526288) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724526465) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94724526489) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31786806619/job/94831186107) |


## [Run #31785237525](https://github.com/sgl-project/sglang/actions/runs/31785237525)
- **分支**: `main`
- **总耗时**: 28.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31785237525

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 27.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31785237525/job/94719688734) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，表明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31785237525/job/94719688734


---
*Auto-generated by npu_pr_monitor.py*