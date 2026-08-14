# NPU CI 执行监控
**生成时间**: 2026-08-14 09:01 UTC
**分析 Run 数**: 64

---

## 📊 本次执行总结

- **成功 Job 数**: 271
- **失败 Run 数**: 62
- **成功 Job 平均耗时**: 25.3min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 131.7min | #31758802795 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525658) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 115.7min | #31762567486 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487197) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 111.5min | #31764589472 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657764258) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.7min | #31761245774 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950829) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.5min | #31759434705 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478699) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.1min | #31762274283 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014950) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.2min | #31760777644 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515944) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.9min | #31758721636 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324946) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.2min | #31762168366 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689380) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.8min | #31761743098 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446488) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31763699808<br>[#16484 Graceful shutdown with SIGTERM for child processes ](https://github.com/sgl-project/sglang/pull/16484) | `feat/graceful-shutdown` | 328.3min | 4 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31763699808) |
| #31764589472<br>[#30398 [Refactor] New EPD](https://github.com/sgl-project/sglang/pull/30398) | `new_epd` | 323.3min | 2 | base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31764589472) |
| #31762567486<br>[#31922 [PD] Prevent outbound ZMQ endpoint cache FD exhaustion](https://github.com/sgl-project/sglang/pull/31922) | `fix/issue-31766-fd-exhaustion` | 322.2min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31762567486) |
| #31765477175<br>[#33684 [Weight Cache] Support static DP/EP layouts](https://github.com/sgl-project/sglang/pull/33684) | `unidy2002/weight-cache-static-dp-ep` | 310.2min | 7 | base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31765477175) |
| #31765141322<br>[#32665 [MoE] Add extension points for custom runner backends](https://github.com/sgl-project/sglang/pull/32665) | `kurt/moe-runner-extension-points` | 306.5min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31765141322) |
| #31765859639<br>[#24959 XPU: Enable GLM5.1 (GlmMoeDsaForCausalLM) DSA Attention](https://github.com/sgl-project/sglang/pull/24959) | `glm5.1_enabling` | 305.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31765859639) |
| #31763732823<br>[#31447 perf: fix overlap scheduling and all-reduce fusion for NVIDIA Confidential Computing(CC) on Blackwell](https://github.com/sgl-project/sglang/pull/31447) | `cc-fixes-rebased` | 293.1min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31763732823) |
| #31763302028<br>[#34796 Add --http2-max-concurrent-streams server arg](https://github.com/sgl-project/sglang/pull/34796) | `cctry/http2-max-concurrent-streams` | 291.6min | 3 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31763302028) |
| #31759434705<br>[#34274 [kernel] Content-addressed JIT build cache, generated from our own ninja](https://github.com/sgl-project/sglang/pull/34274) | `jit-content-addressed-cache` | 277.4min | 2 | base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31759434705) |
| #31762274283<br>[#31794 [AMD][Fix] Qwen3.5: guard zero-grid launch in fused_qk_gemma_rmsnorm(_with_gate) (HIP invalid configuration on idle DP rank)](https://github.com/sgl-project/sglang/pull/31794) | `fix/qwen35-fused-qk-rmsnorm-zerogrid-31350` | 260.5min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31762274283) |
| #31762448440<br>[#34560 [Fix] Fix Qwen3.5 MTP startup with HiCache](https://github.com/sgl-project/sglang/pull/34560) | `fix/qwen35-hicache-mtp-draft-depth` | 253.0min | 3 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31762448440) |
| #31762170839<br>[#34498 [ROCm] Direct-write a8w8 bmm output to eliminate o_proj transpose copy](https://github.com/sgl-project/sglang/pull/34498) | `opt/kimi-k2-mxfp4-fp8-bmm-direct-write` | 230.9min | 5 | base-b-test-4-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31762170839) |
| #31762168366<br>[#34502 [ROCm] Fuse per-token fp8 activation quant into RMSNorm for per-chann…](https://github.com/sgl-project/sglang/pull/34502) | `opt/kimi-k2-mxfp4-fuse-pertoken-quant` | 226.5min | 4 | base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31762168366) |
| #31758721636<br>[#32405 [MoE Refactor] Migrate SM100 trtllm-gen mxfp4 MoE onto MoeRunner](https://github.com/sgl-project/sglang/pull/32405) | `refactor-mxfp4-sm100-trtllm-moerunner` | 210.0min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31758721636) |
| #31766134411<br>[#33354 [XPU] Use a fused GDN kernel from sgl-kernel for Qwen3.5](https://github.com/sgl-project/sglang/pull/33354) | `qwen3.5_gdn_xpu_kernel` | 204.2min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31766134411) |
| #31761245774<br>[#32597 Support streaming session on NPU](https://github.com/sgl-project/sglang/pull/32597) | `streaming_session` | 203.7min | 5 | base-b-test-16-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31761245774) |
| #31758298364<br>[#29328 [AMD][Quantization] Online MXFP4 quantization 4/N - NVFP4 to MXFP4 Online Requantization on AMD GPUs](https://github.com/sgl-project/sglang/pull/29328) | `online-nvfp4-to-mxfp4-convert` | 193.7min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31758298364) |
| #31760777644<br>[#33685 [NPU CI] Reorganize test output/log directory structure with workflow context](https://github.com/sgl-project/sglang/pull/33685) | `pllimax/output-log-dir-structure` | 193.0min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31760777644) |
| #31761589073<br>[#34753 feat(cli): add extensible serve backend plugins](https://github.com/sgl-project/sglang/pull/34753) | `codex/extensible-serve-backends` | 192.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31761589073) |
| #31761743098<br>[#34284 fix(scheduler): track max prefill batch size over recent real admissions](https://github.com/sgl-project/sglang/pull/34284) | `real_max_prefill_size` | 191.4min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31761743098) |
| #31758217869<br>[#34774 [Fix] has_hf_quant_config crashes on local dirs without the config](https://github.com/sgl-project/sglang/pull/34774) | `fix/has-hf-quant-config-local-dirs` | 172.8min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31758217869) |
| #31758802795<br>[#34432 [AMD][DCP 1/N] add dcp support for aiter backend](https://github.com/sgl-project/sglang/pull/34432) | `k3_dcp_1n` | 151.0min | 5 | multimodal-gen-test-1-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31758802795) |
| #31772016322<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `codex/k3-mm-cache-k3` | 133.8min | 10 | base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31772016322) |
| #31764399793 | `fuse-swiglu-moe-up-gemm-epilogue` | 127.0min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31764399793) |
| #31770907817 | `fuse-swiglu-moe-up-gemm-epilogue` | 105.1min | 10 | base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31770907817) |
| #31767269634<br>[#33676 [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management](https://github.com/sgl-project/sglang/pull/33676) | `main_8.5` | 84.9min | 11 | base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31767269634) |
| #31766023771<br>[#34331 [quantization] Add tuned Triton tile configs for channelwise FP8 GEMM…](https://github.com/sgl-project/sglang/pull/34331) | `main` | 75.8min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31766023771) |
| #31765230258<br>[#34801 [PD] Preserve decode KV across retraction in HiCache](https://github.com/sgl-project/sglang/pull/34801) | `shiyang/pd-host-pool-retraction-backup` | 74.2min | 10 | base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31765230258) |
| #31777622204<br>[#34736 [Diffusion] Unify component residency controls](https://github.com/sgl-project/sglang/pull/34736) | `codex/component-residency-policy` | 73.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31777622204) |
| #31770181418<br>[#28666 [AMD] Fuse shared_expert_gate GEMV into the MoE append kernel (HIP/aiter)](https://github.com/sgl-project/sglang/pull/28666) | `main` | 68.8min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31770181418) |
| #31762212381<br>[#33684 [Weight Cache] Support static DP/EP layouts](https://github.com/sgl-project/sglang/pull/33684) | `unidy2002/weight-cache-static-dp-ep` | 63.9min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31762212381) |
| #31773774563<br>[#34496 Fix eager AMX backend probe imports](https://github.com/sgl-project/sglang/pull/34496) | `datdo/lazy-amx-backend-probe` | 63.7min | 11 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31773774563) |
| #31759197901<br>[#34592 [GDN] Honor configured linear-attn verify backend in the kernel dispatcher](https://github.com/sgl-project/sglang/pull/34592) | `main` | 60.8min | 10 | base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31759197901) |
| #31759311404<br>[#25855 perf(jit_kernel/deepseek_v4): optimize paged_mqa_metadata](https://github.com/sgl-project/sglang/pull/25855) | `feature/optimize-paged-mqa-metadata` | 58.8min | 10 | base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31759311404) |
| #31771892853<br>[#34820 draft test](https://github.com/sgl-project/sglang/pull/34820) | `align-mamba-checkpoint-grid` | 58.1min | 10 | base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31771892853) |
| #31760509867<br>[#33922 Fix Qwen3.5 GDN multi-item scoring](https://github.com/sgl-project/sglang/pull/33922) | `fix/qwen35-gdn-mis` | 55.4min | 10 | base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31760509867) |
| #31772550628<br>[#34736 [Diffusion] Unify component residency controls](https://github.com/sgl-project/sglang/pull/34736) | `codex/component-residency-policy` | 48.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31772550628) |
| #31774687575<br>[#34753 feat(cli): add extensible serve backend plugins](https://github.com/sgl-project/sglang/pull/34753) | `main` | 44.6min | 10 | base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31774687575) |
| #31777303568<br>[#34820 draft test](https://github.com/sgl-project/sglang/pull/34820) | `align-mamba-checkpoint-grid` | 42.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31777303568) |
| #31770120675 | `feat/kv-events-component-placement-v2` | 41.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31770120675) |
| #31764978526<br>[#34736 [Diffusion] Unify component residency controls](https://github.com/sgl-project/sglang/pull/34736) | `codex/component-residency-policy` | 40.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31764978526) |
| #31763025422<br>[#34736 [Diffusion] Unify component residency controls](https://github.com/sgl-project/sglang/pull/34736) | `codex/component-residency-policy` | 38.9min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31763025422) |
| #31775345351<br>[#34820 draft test](https://github.com/sgl-project/sglang/pull/34820) | `align-mamba-checkpoint-grid` | 34.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31775345351) |
| #31762381721<br>[#25855 perf(jit_kernel/deepseek_v4): optimize paged_mqa_metadata](https://github.com/sgl-project/sglang/pull/25855) | `main` | 27.0min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31762381721) |
| #31770407266 | `kda_fused_accept_state` | 25.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31770407266) |
| #31769067703<br>[#34801 [PD] Preserve decode KV across retraction in HiCache](https://github.com/sgl-project/sglang/pull/34801) | `shiyang/pd-host-pool-retraction-backup` | 25.1min | 11 | base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31769067703) |
| #31763861002<br>[#34309 [CI] Prune redundant CPU test overhead](https://github.com/sgl-project/sglang/pull/34309) | `xinyuan/cpu-ci-prune-static-overhead` | 22.8min | 12 | base-b-test-16-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31763861002) |
| #31764988709<br>[#34309 [CI] Prune redundant CPU test overhead](https://github.com/sgl-project/sglang/pull/34309) | `main` | 20.6min | 12 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31764988709) |
| #31768358794<br>[#31768 [Model] Add LLaDA2.2 Block Routing MoE support](https://github.com/sgl-project/sglang/pull/31768) | `feat/llada2-block-routing` | 20.4min | 12 | base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), multimodal-gen-test-1-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31768358794) |
| #31779594768<br>[#34404 [VLM] Cache Kimi-K3 per-image processor artifacts](https://github.com/sgl-project/sglang/pull/34404) | `codex/k3-mm-cache-k3` | 19.8min | 11 | base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31779594768) |
| #31779378860<br>[#34774 [Fix] has_hf_quant_config crashes on local dirs without the config](https://github.com/sgl-project/sglang/pull/34774) | `main` | 16.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31779378860) |
| #31762265214<br>[#34299 [KDA] Close Phase A CAKE engagement and zero-copy admission](https://github.com/sgl-project/sglang/pull/34299) | `codex/sglang-phase-a-admission-rebased-20260810` | 14.9min | 12 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31762265214) |
| #31780412900<br>[#34650 feat(diffusion): rebuild MiniMax-H3 AdaLN outputs on demand](https://github.com/sgl-project/sglang/pull/34650) | `main` | 13.4min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31780412900) |
| #31777850096<br>[#34121 [Diffusion] Fix cache-first fast path accepting a metadata-only snapshot](https://github.com/sgl-project/sglang/pull/34121) | `main` | 13.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31777850096) |
| #31773782306<br>[#34274 [kernel] Content-addressed JIT build cache, generated from our own ninja](https://github.com/sgl-project/sglang/pull/34274) | `main` | 13.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31773782306) |
| #31776525783 | `cheng/gc-sr-review` | 12.4min | 11 | base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31776525783) |
| #31778624352<br>[#34741 [AMD] Fix Triton 3.7 gfx950 extend-attention spills](https://github.com/sgl-project/sglang/pull/34741) | `main` | 12.2min | 10 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31778624352) |
| #31777336383<br>[#34496 Fix eager AMX backend probe imports](https://github.com/sgl-project/sglang/pull/34496) | `main` | 8.9min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31777336383) |
| #31757764626<br>[#28354 [FlashInfer v0.6.16] Support FlashInfer CuTe DSL NVFP4 MoE quantization](https://github.com/sgl-project/sglang/pull/28354) | `main` | 7.9min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31757764626) |
| #31779606748 | `lm-head-opt` | 7.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31779606748) |
| #31774987068<br>[#34820 draft test](https://github.com/sgl-project/sglang/pull/34820) | `align-mamba-checkpoint-grid` | 6.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31774987068) |
| #31769889807<br>[#34788 [Fix] Restore layer-level DSV4 RoPE policy](https://github.com/sgl-project/sglang/pull/34788) | `main` | 6.0min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31769889807) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 56 | 49 | 0 | 7 |
| base-b-test-1-npu-a3 / run (0) | 56 | 19 | 0 | 37 |
| base-b-test-16-npu-a3 / run (0) | 56 | 17 | 0 | 39 |
| base-b-test-2-npu-a3 / run (0) | 56 | 19 | 0 | 37 |
| base-b-test-4-npu-a3 / run (0) | 56 | 17 | 0 | 39 |
| base-b-test-4-npu-a3 / run (1) | 56 | 18 | 0 | 38 |
| base-b-test-8-npu-a3 / run (0) | 56 | 19 | 0 | 37 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 56 | 18 | 0 | 38 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 56 | 15 | 0 | 41 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 56 | 18 | 0 | 38 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 56 | 21 | 0 | 35 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 18 | 0 | 0 | 18 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 15 | 2 | 0 | 13 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 18 | 5 | 0 | 13 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21 | 3 | 0 | 18 |
| multimodal-gen-test-1-npu-a3 | 58 | 31 | 5 | 22 |

---


## [Run #31780412900](https://github.com/sgl-project/sglang/actions/runs/31780412900)
- **分支**: `main`
- **总耗时**: 13.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31780412900

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31780412900/job/94704890457) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的输出，仅有GitHub Actions环境准备、Node版本警告及上传失败产物（无文件）等常规信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31780412900/job/94704890457


## [Run #31779606748](https://github.com/sgl-project/sglang/actions/runs/31779606748)
- **分支**: `lm-head-opt`
- **总耗时**: 7.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31779606748

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374231) |
| base-b-test-2-npu-a3 / run (0) | 6.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374335) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374357) |
| base-b-test-4-npu-a3 / run (0) | 6.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374370) |
| base-b-test-4-npu-a3 / run (1) | 6.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374404) |
| base-b-test-16-npu-a3 / run (0) | 6.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374424) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374549) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374572) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374575) |
| base-b-test-1-npu-a3 / run (0) | 6.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702375092) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702375098) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体错误或失败断言，仅显示Node 20弃用警告和上传artifact时未找到文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374231

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374335

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374357

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或资源管理问题，需检查存储路径或重新上传产物。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374370

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374404

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374424

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖或缓存文件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374549

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374572

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374575

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702375092

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702375098

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31779606748/job/94702374362) |


## [Run #31779594768](https://github.com/sgl-project/sglang/actions/runs/31779594768)
- **分支**: `codex/k3-mm-cache-k3`
- **总耗时**: 19.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31779594768

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363857) |
| base-b-test-2-npu-a3 / run (0) | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363865) |
| base-b-test-4-npu-a3 / run (0) | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363903) |
| base-b-test-4-npu-a3 / run (1) | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363906) |
| multimodal-gen-test-1-npu-a3 | 18.6min | 其他 | 作业日志不完整，未显示实际测试命令和失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363923) |
| base-b-test-1-npu-a3 / run (0) | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363986) |
| base-b-test-16-npu-a3 / run (0) | 18.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363990) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702364142) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702364177) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702364203) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 18.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702364223) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363857

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363865

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363903

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明作业依赖的某个文件或数据在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查 CI 配置中的 blob 引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363906

- **multimodal-gen-test-1-npu-a3**: 日志中未包含multimodal-gen-test的实际执行输出，只有checkout、upload-artifact等基础设施步骤，且upload-artifact提示无文件上传。无法判断具体失败原因，可能是测试未运行或日志被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363923

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363986

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363990

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径错误，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702364142

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702364177

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702364203

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702364223

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31779594768/job/94702363886) |


## [Run #31779378860](https://github.com/sgl-project/sglang/actions/runs/31779378860)
- **分支**: `main`
- **总耗时**: 16.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31779378860

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691575) |
| base-b-test-2-npu-a3 / run (0) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691582) |
| base-b-test-4-npu-a3 / run (1) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691695) |
| base-b-test-1-npu-a3 / run (0) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691711) |
| base-b-test-8-npu-a3 / run (0) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691750) |
| base-b-test-16-npu-a3 / run (0) | 15.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691774) |
| base-b-test-4-npu-a3 / run (0) | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691775) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701692110) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701692153) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701692188) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 15.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701692229) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败输出，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691575

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691582

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691695

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691711

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于环境或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691750

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691774

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691775

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或依赖文件在 Azure Blob 存储中已被删除或路径错误，属于环境配置或资源缺失问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701692110

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在存储中不存在，可能是文件被删除、路径错误或上传未完成，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701692153

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701692188

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的构建产物或缓存文件未上传或已被删除，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701692229

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31779378860/job/94701691637) |


## [Run #31778624352](https://github.com/sgl-project/sglang/actions/runs/31778624352)
- **分支**: `main`
- **总耗时**: 12.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31778624352

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 11.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399792) |
| base-b-test-1-npu-a3 / run (0) | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399849) |
| base-b-test-4-npu-a3 / run (1) | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399931) |
| base-b-test-4-npu-a3 / run (0) | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399932) |
| base-b-test-16-npu-a3 / run (0) | 11.6min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399968) |
| base-b-test-8-npu-a3 / run (0) | 11.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400006) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400215) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400231) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400264) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400298) |

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问 Azure Blob 中的某个 blob，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误导致，属于基础设施或环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399792

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是上游构建未成功上传或存储配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399849

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于环境或资源配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399931

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399932

- **base-b-test-16-npu-a3 / run (0)**: 作业日志显示BlobNotFound错误，说明CI流程尝试下载的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399968

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400006

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400215

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400231

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400264

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存、依赖或日志文件未正确上传，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699400298

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31778624352/job/94699399910) |


## [Run #31777850096](https://github.com/sgl-project/sglang/actions/runs/31777850096)
- **分支**: `main`
- **总耗时**: 13.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31777850096

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777850096/job/94697058685) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777850096/job/94697058685


## [Run #31777622204](https://github.com/sgl-project/sglang/actions/runs/31777622204)
- **分支**: `codex/component-residency-policy`
- **总耗时**: 73.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31777622204

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 54.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777622204/job/94696349924) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示任何测试执行或失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777622204/job/94696349924


## [Run #31777336383](https://github.com/sgl-project/sglang/actions/runs/31777336383)
- **分支**: `main`
- **总耗时**: 8.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31777336383

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548821) |
| base-b-test-16-npu-a3 / run (0) | 8.0min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取所需数据或日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548895) |
| base-b-test-2-npu-a3 / run (0) | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548911) |
| base-b-test-1-npu-a3 / run (0) | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548951) |
| base-b-test-4-npu-a3 / run (0) | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548958) |
| base-b-test-8-npu-a3 / run (0) | 8.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548983) |
| base-b-test-4-npu-a3 / run (1) | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549029) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549190) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549194) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549206) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549256) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是由于资源被清理、上传失败或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548821

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明指定的blob在存储账户中不存在，可能是文件被删除、路径错误或上传失败。这属于外部依赖资源缺失，非代码或性能问题，需检查CI配置中的blob路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548895

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548911

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548951

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548958

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695548983

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549029

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549190

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549194

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549206

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549256

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31777336383/job/94695549065) |


## [Run #31777303568](https://github.com/sgl-project/sglang/actions/runs/31777303568)
- **分支**: `align-mamba-checkpoint-grid`
- **总耗时**: 42.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31777303568

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492759) |
| base-b-test-4-npu-a3 / run (1) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492796) |
| base-b-test-4-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492806) |
| base-b-test-8-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492822) |
| base-b-test-16-npu-a3 / run (0) | 41.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492842) |
| base-b-test-1-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492846) |
| base-b-test-2-npu-a3 / run (0) | 41.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492950) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695493124) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695493164) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695493168) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 41.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695493264) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492759

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492796

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492806

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492822

- **base-b-test-16-npu-a3 / run (0)**: 作业失败原因是访问Azure Blob存储时返回BlobNotFound错误，即请求的资源不存在。这通常是由于日志或工件已被清理、路径错误或存储配置变更所致，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492842

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492846

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492950

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695493124

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695493164

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695493168

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695493264

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31777303568/job/94695492918) |


## [Run #31776525783](https://github.com/sgl-project/sglang/actions/runs/31776525783)
- **分支**: `cheng/gc-sr-review`
- **总耗时**: 12.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31776525783

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069518) |
| multimodal-gen-test-1-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069539) |
| base-b-test-2-npu-a3 / run (0) | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069596) |
| base-b-test-16-npu-a3 / run (0) | 11.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069599) |
| base-b-test-8-npu-a3 / run (0) | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069681) |
| base-b-test-4-npu-a3 / run (0) | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069709) |
| base-b-test-4-npu-a3 / run (1) | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069712) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693070009) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693070063) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693070110) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693070144) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069518

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069539

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069596

- **base-b-test-16-npu-a3 / run (0)**: 作业失败原因是BlobNotFound错误，即CI系统尝试下载或访问的Azure Blob存储对象不存在，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069599

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069681

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069709

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069712

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693070009

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693070063

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693070110

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693070144

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31776525783/job/94693069725) |


## [Run #31775345351](https://github.com/sgl-project/sglang/actions/runs/31775345351)
- **分支**: `align-mamba-checkpoint-grid`
- **总耗时**: 34.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31775345351

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625516) |
| base-b-test-2-npu-a3 / run (0) | 33.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625593) |
| base-b-test-8-npu-a3 / run (0) | 33.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625646) |
| base-b-test-16-npu-a3 / run (0) | 33.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625666) |
| base-b-test-1-npu-a3 / run (0) | 33.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625691) |
| base-b-test-4-npu-a3 / run (0) | 33.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625736) |
| base-b-test-4-npu-a3 / run (1) | 33.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625737) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 33.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625939) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 33.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625979) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 33.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625992) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 33.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625997) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有runner启动、依赖下载、上传artifact（无文件）及清理步骤。无法判断失败原因，可能是测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625516

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传，或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625593

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625646

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是日志上传或依赖下载路径错误，属于基础设施或配置问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625666

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625691

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625736

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625737

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、缓存或构建产物）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625939

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625979

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625992

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625997

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31775345351/job/94689625693) |


## [Run #31774987068](https://github.com/sgl-project/sglang/actions/runs/31774987068)
- **分支**: `align-mamba-checkpoint-grid`
- **总耗时**: 6.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31774987068

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537412) |
| base-b-test-4-npu-a3 / run (1) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537435) |
| base-b-test-8-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537453) |
| base-b-test-4-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537491) |
| base-b-test-2-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537498) |
| base-b-test-16-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537501) |
| base-b-test-1-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537513) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537657) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537719) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537740) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537815) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537412

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是上游产物未上传或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537435

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537453

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537491

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537498

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537501

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537513

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537657

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537719

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537740

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537815

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31774987068/job/94688537407) |


## [Run #31774774812](https://github.com/sgl-project/sglang/actions/runs/31774774812)
- **分支**: `adaln-online`
- **总耗时**: 60.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31774774812

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 43.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31774774812/job/94687903773) |


## [Run #31774687575](https://github.com/sgl-project/sglang/actions/runs/31774687575)
- **分支**: `main`
- **总耗时**: 44.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31774687575

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 44.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642485) |
| base-b-test-16-npu-a3 / run (0) | 44.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642532) |
| base-b-test-2-npu-a3 / run (0) | 44.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642547) |
| base-b-test-4-npu-a3 / run (0) | 44.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642548) |
| base-b-test-1-npu-a3 / run (0) | 44.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642557) |
| base-b-test-4-npu-a3 / run (1) | 44.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642645) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 44.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642697) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642701) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 44.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642729) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 44.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642750) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642485

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载的构建产物或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642532

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642547

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642548

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642557

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642645

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物或依赖未正确上传，或路径配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642697

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642701

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径和生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642729

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642750

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31774687575/job/94687642497) |


## [Run #31773782306](https://github.com/sgl-project/sglang/actions/runs/31773782306)
- **分支**: `main`
- **总耗时**: 13.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31773782306

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684965814) |
| base-b-test-16-npu-a3 / run (0) | 12.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966129) |
| base-b-test-4-npu-a3 / run (1) | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966186) |
| base-b-test-1-npu-a3 / run (0) | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966233) |
| base-b-test-4-npu-a3 / run (0) | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966238) |
| base-b-test-8-npu-a3 / run (0) | 12.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966329) |
| base-b-test-2-npu-a3 / run (0) | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966446) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684967357) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684967387) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684967389) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684967413) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684965814

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966129

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966186

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966233

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966238

- **base-b-test-8-npu-a3 / run (0)**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是资源被删除、路径错误或上传未完成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966329

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966446

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是构建产物未上传或存储配置变更，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684967357

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，可能是 CI 依赖的预构建产物或缓存文件未上传或已被删除，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684967387

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684967389

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重、测试数据或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684967413

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31773782306/job/94684966203) |


## [Run #31773774563](https://github.com/sgl-project/sglang/actions/runs/31773774563)
- **分支**: `datdo/lazy-amx-backend-probe`
- **总耗时**: 63.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31773774563

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932035) |
| base-b-test-1-npu-a3 / run (0) | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932046) |
| base-a-test-1-npu-a2 / run (0) | 1.9min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932053) |
| base-b-test-4-npu-a3 / run (0) | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932056) |
| base-b-test-4-npu-a3 / run (1) | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932067) |
| base-b-test-8-npu-a3 / run (0) | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932081) |
| base-b-test-16-npu-a3 / run (0) | 63.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932089) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932301) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932321) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932374) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 63.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932463) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932035

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是缓存、依赖或上传步骤出现问题，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932046

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 工具链时，从内部缓存服务下载 rust-1.92 通道文件超时，导致脚本执行失败，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932053

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932056

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932067

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932081

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932089

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932301

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932321

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932374

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到。这可能是由于 CI 配置中引用的 blob 路径错误、blob 被删除或未正确上传，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684932463

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 35.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31773774563/job/94684931801) |


## [Run #31772550628](https://github.com/sgl-project/sglang/actions/runs/31772550628)
- **分支**: `codex/component-residency-policy`
- **总耗时**: 48.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31772550628

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772550628/job/94681339284) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures工件时未找到文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772550628/job/94681339284


## [Run #31772016322](https://github.com/sgl-project/sglang/actions/runs/31772016322)
- **分支**: `codex/k3-mm-cache-k3`
- **总耗时**: 133.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31772016322

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776286) |
| base-b-test-16-npu-a3 / run (0) | 133.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776349) |
| base-b-test-4-npu-a3 / run (0) | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776356) |
| base-b-test-4-npu-a3 / run (1) | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776411) |
| base-b-test-1-npu-a3 / run (0) | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776463) |
| base-b-test-2-npu-a3 / run (0) | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776466) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776615) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776628) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776685) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 133.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776761) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776286

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776349

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776356

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776411

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776463

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776466

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776615

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776628

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776685

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776761

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776289) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31772016322/job/94679776309) |


## [Run #31771892853](https://github.com/sgl-project/sglang/actions/runs/31771892853)
- **分支**: `align-mamba-checkpoint-grid`
- **总耗时**: 58.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31771892853

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 57.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426613) |
| base-b-test-2-npu-a3 / run (0) | 57.7min | 环境问题 | Azure Blob 存储中指定的工件不存在，导致作业无法下载依赖。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426635) |
| base-b-test-8-npu-a3 / run (0) | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426641) |
| base-b-test-4-npu-a3 / run (0) | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426672) |
| base-b-test-1-npu-a3 / run (0) | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426681) |
| base-b-test-4-npu-a3 / run (1) | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426705) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426936) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426938) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426957) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 57.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426962) |

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误所致，属于基础设施或环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426613

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试获取的 blob 已被删除或路径错误，可能是上游构建未成功上传工件或存储配置问题，需检查相关工件生成与上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426635

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426641

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426672

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426681

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426705

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426936

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426938

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426957

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426962

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426537) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31771892853/job/94679426614) |


## [Run #31770907817](https://github.com/sgl-project/sglang/actions/runs/31770907817)
- **分支**: `fuse-swiglu-moe-up-gemm-epilogue`
- **总耗时**: 105.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31770907817

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 104.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534665) |
| base-b-test-2-npu-a3 / run (0) | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534682) |
| base-b-test-4-npu-a3 / run (1) | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534704) |
| base-b-test-8-npu-a3 / run (0) | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534722) |
| base-b-test-4-npu-a3 / run (0) | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534737) |
| base-b-test-1-npu-a3 / run (0) | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534823) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534889) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534892) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534906) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 104.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534938) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534665

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534682

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534704

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534722

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534737

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534823

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534889

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534892

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534906

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534938

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534659) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770907817/job/94676534739) |


## [Run #31770407266](https://github.com/sgl-project/sglang/actions/runs/31770407266)
- **分支**: `kda_fused_accept_state`
- **总耗时**: 25.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31770407266

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 19.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675008967) |
| base-b-test-1-npu-a3 / run (0) | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675008999) |
| base-b-test-8-npu-a3 / run (0) | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009006) |
| base-b-test-4-npu-a3 / run (1) | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009007) |
| base-b-test-16-npu-a3 / run (0) | 24.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009034) |
| base-b-test-2-npu-a3 / run (0) | 24.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009048) |
| base-b-test-4-npu-a3 / run (0) | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009050) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009160) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009165) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009177) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 24.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009215) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675008967

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675008999

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009006

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009007

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问的 blob 资源未找到（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009034

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009048

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径错误，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009050

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009160

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009165

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009177

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009215

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770407266/job/94675009026) |


## [Run #31770181418](https://github.com/sgl-project/sglang/actions/runs/31770181418)
- **分支**: `main`
- **总耗时**: 68.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31770181418

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389735) |
| base-b-test-4-npu-a3 / run (1) | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389757) |
| base-b-test-2-npu-a3 / run (0) | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389765) |
| base-b-test-4-npu-a3 / run (0) | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389787) |
| base-b-test-16-npu-a3 / run (0) | 68.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389812) |
| base-b-test-8-npu-a3 / run (0) | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389888) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674390054) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674390167) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674390199) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 68.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674390306) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389735

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389757

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389765

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389787

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象缺失，可能是构建产物未上传、路径错误或存储被清理，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389812

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389888

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径错误，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674390054

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674390167

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674390199

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674390306

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389738) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770181418/job/94674389850) |


## [Run #31770120675](https://github.com/sgl-project/sglang/actions/runs/31770120675)
- **分支**: `feat/kv-events-component-placement-v2`
- **总耗时**: 41.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31770120675

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 22.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674161961) |
| base-b-test-1-npu-a3 / run (0) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162003) |
| base-b-test-16-npu-a3 / run (0) | 40.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162048) |
| base-b-test-2-npu-a3 / run (0) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162146) |
| base-b-test-8-npu-a3 / run (0) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162150) |
| base-b-test-4-npu-a3 / run (1) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162181) |
| base-b-test-4-npu-a3 / run (0) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162199) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162327) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162353) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162357) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162463) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node版本警告及上传diffusion-failures工件时未找到文件，未包含multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674161961

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162003

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162048

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162146

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162150

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162181

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162199

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162327

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162353

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162357

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674162463

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31770120675/job/94674161992) |


## [Run #31769889807](https://github.com/sgl-project/sglang/actions/runs/31769889807)
- **分支**: `main`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31769889807

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500687) |
| base-b-test-16-npu-a3 / run (0) | 5.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500736) |
| base-b-test-2-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500737) |
| base-b-test-8-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500749) |
| base-b-test-4-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500777) |
| base-b-test-4-npu-a3 / run (1) | 5.3min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500779) |
| base-a-test-1-npu-a2 / run (0) | 3.9min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500797) |
| base-b-test-1-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500827) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500912) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500932) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500985) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673501036) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或文件被清理，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500687

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查CI配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500736

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500737

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500749

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500777

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于环境/资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500779

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在构建xatlas依赖时，自定义容器实现执行失败，错误信息为“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500797

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500827

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500912

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500932

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673500985

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769889807/job/94673501036


## [Run #31769067703](https://github.com/sgl-project/sglang/actions/runs/31769067703)
- **分支**: `shiyang/pd-host-pool-retraction-backup`
- **总耗时**: 25.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31769067703

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041667) |
| base-b-test-4-npu-a3 / run (1) | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041672) |
| base-b-test-8-npu-a3 / run (0) | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041695) |
| base-b-test-1-npu-a3 / run (0) | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041701) |
| multimodal-gen-test-1-npu-a3 | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041704) |
| base-b-test-16-npu-a3 / run (0) | 24.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041733) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041790) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041793) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041803) |
| base-b-test-2-npu-a3 / run (0) | 24.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041819) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041828) |

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的构建产物或依赖文件在 Azure Blob 存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041667

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041672

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041695

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041701

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041704

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问的 blob 资源缺失（BlobNotFound），可能是日志上传延迟、路径错误或文件被清理，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041733

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041790

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041793

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个远程资源（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，属于环境配置或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041803

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041819

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖或缓存文件在 Azure Blob 存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041828

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31769067703/job/94671041722) |


## [Run #31768358794](https://github.com/sgl-project/sglang/actions/runs/31768358794)
- **分支**: `feat/llada2-block-routing`
- **总耗时**: 20.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31768358794

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 19.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905859) |
| base-b-test-1-npu-a3 / run (0) | 19.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905861) |
| base-b-test-8-npu-a3 / run (0) | 19.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905869) |
| base-b-test-4-npu-a3 / run (0) | 19.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905874) |
| base-b-test-4-npu-a3 / run (1) | 19.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905876) |
| base-b-test-2-npu-a3 / run (0) | 19.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905891) |
| base-a-test-1-npu-a2 / run (0) | 19.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905930) |
| multimodal-gen-test-1-npu-a3 | 14.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905938) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668906067) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668906125) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668906146) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668906163) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905859

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是缓存或依赖文件未正确上传，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905861

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905869

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是缓存、依赖或上传步骤异常，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905874

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905876

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905891

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905930

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅有Node.js版本弃用警告和上传artifact时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668905938

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668906067

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668906125

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668906146

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31768358794/job/94668906163


## [Run #31767269634](https://github.com/sgl-project/sglang/actions/runs/31767269634)
- **分支**: `main_8.5`
- **总耗时**: 84.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31767269634

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 0.9min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659203) |
| base-b-test-4-npu-a3 / run (0) | 84.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659209) |
| base-b-test-4-npu-a3 / run (1) | 84.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659234) |
| base-b-test-1-npu-a3 / run (0) | 84.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659243) |
| base-b-test-8-npu-a3 / run (0) | 84.3min | 环境问题 | CI作业因Azure Blob存储中指定工件不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659252) |
| base-b-test-2-npu-a3 / run (0) | 84.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659268) |
| base-b-test-16-npu-a3 / run (0) | 84.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659273) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659504) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 84.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659515) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 84.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659592) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 84.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659769) |

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发fast-fail机制，作业被终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659203

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源过期或配置问题，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659209

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659234

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659243

- **base-b-test-8-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业尝试下载的构建产物或缓存文件在存储中缺失，可能是上游作业未成功上传或路径配置错误，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659252

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659268

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该文件不存在（BlobNotFound）。这通常是日志上传失败、路径错误或存储被清理所致，属于基础设施或配置问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659273

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659504

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659515

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659592

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659769

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31767269634/job/94665659189) |


## [Run #31766407426](https://github.com/sgl-project/sglang/actions/runs/31766407426)
- **分支**: `h3_xpu_support`
- **总耗时**: 36.0min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31766407426

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766407426/job/94663211771) |


## [Run #31766134411](https://github.com/sgl-project/sglang/actions/runs/31766134411)
- **分支**: `qwen3.5_gdn_xpu_kernel`
- **总耗时**: 204.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31766134411

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391744) |
| base-b-test-4-npu-a3 / run (1) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391758) |
| base-b-test-8-npu-a3 / run (0) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391809) |
| base-b-test-1-npu-a3 / run (0) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391830) |
| base-b-test-4-npu-a3 / run (0) | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391848) |
| base-b-test-16-npu-a3 / run (0) | 203.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391886) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662392056) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662392060) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662392070) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 203.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662392081) |

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391744

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391758

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391809

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391830

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是构建产物未上传或存储配置变更，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391848

- **base-b-test-16-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，表明CI系统尝试访问的Azure Blob存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391886

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662392056

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662392060

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662392070

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662392081

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391711) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766134411/job/94662391731) |


## [Run #31766023771](https://github.com/sgl-project/sglang/actions/runs/31766023771)
- **分支**: `main`
- **总耗时**: 75.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31766023771

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194682) |
| base-b-test-16-npu-a3 / run (0) | 74.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194695) |
| base-b-test-2-npu-a3 / run (0) | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194759) |
| base-b-test-4-npu-a3 / run (0) | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194819) |
| base-b-test-4-npu-a3 / run (1) | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194863) |
| base-b-test-8-npu-a3 / run (0) | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194881) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194916) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194967) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194989) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 74.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662195015) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194682

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志被清理、路径错误或上传失败，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194695

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194759

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194819

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194863

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194881

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194916

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194967

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194989

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662195015

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194703) |
| multimodal-gen-test-1-npu-a3 | 26.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31766023771/job/94662194737) |


## [Run #31765859639](https://github.com/sgl-project/sglang/actions/runs/31765859639)
- **分支**: `glm5.1_enabling`
- **总耗时**: 305.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31765859639

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 35.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553403) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553494) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553496) |
| base-b-test-16-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553531) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553589) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553599) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553634) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553785) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.9min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553825) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 作业因健康检查被快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553876) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | PR测试健康检查失败，根因是其他作业失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553891) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时未找到文件，以及Node 20弃用警告。无法判断具体失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553403

- **base-b-test-1-npu-a3 / run (0)**: 该作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，因此主动跳过执行，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553494

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，因此本作业（base-b-test-2-npu-a3）被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553496

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-16-npu-a3）被快速失败跳过，并非自身执行出错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553531

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤，随后触发fast-fail跳过执行，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553589

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业因快速失败机制被终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553599

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553634

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因作业为multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553785

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553825

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业被识别为级联失败并过滤，实际未执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553876

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553891

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765859639/job/94661553562) |


## [Run #31765477175](https://github.com/sgl-project/sglang/actions/runs/31765477175)
- **分支**: `unidy2002/weight-cache-static-dp-ep`
- **总耗时**: 310.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31765477175

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491722) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491768) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491808) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 2.5min | 环境问题 | rustup 下载超时导致环境初始化失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660492225) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660492255) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 2.4min | 环境问题 | rustup 下载超时导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660492262) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94709119884) |

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业base-c-test-acc-8-npu-a3，本作业因快速失败策略被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491722

- **base-b-test-4-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因作业base-c-test-acc-8-npu-a3失败，触发fast-fail机制，本作业未实际运行测试即被取消，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491768

- **base-b-test-16-npu-a3 / run (0)**: 日志显示health-check检测到base-c-test-acc-8-npu-a3作业失败，被判定为根因失败，因此本作业（base-b-test-16-npu-a3）被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491808

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在安装 Rust 工具链时，从内部缓存服务下载 rust-1.92 通道文件超时，导致脚本退出码非零，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660492225

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到base-c-test-acc-8-npu-a3作业失败，被判定为根因作业，因此本作业（base-c-test-acc-4-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660492255

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 在安装 Rust 工具链时，从内部缓存服务下载 rustup 元数据文件超时，导致安装失败，作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660492262

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-acc-2-npu-a3和base-c-test-acc-8-npu-a3两个根因失败作业，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94709119884

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 33.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491471) |
| base-b-test-1-npu-a3 / run (0) | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491660) |
| base-a-test-1-npu-a2 / run (0) | 7.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491724) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491748) |
| base-b-test-2-npu-a3 / run (0) | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660491847) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765477175/job/94660492261) |


## [Run #31765230258](https://github.com/sgl-project/sglang/actions/runs/31765230258)
- **分支**: `shiyang/pd-host-pool-retraction-backup`
- **总耗时**: 74.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31765230258

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 73.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767554) |
| base-b-test-2-npu-a3 / run (0) | 73.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767625) |
| base-b-test-1-npu-a3 / run (0) | 73.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767630) |
| base-b-test-8-npu-a3 / run (0) | 73.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767664) |
| base-b-test-4-npu-a3 / run (0) | 73.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767692) |
| base-b-test-4-npu-a3 / run (1) | 73.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767696) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 73.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767808) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 73.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767827) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 73.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767854) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 73.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767859) |

- **base-b-test-16-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767554

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767625

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767630

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767664

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767692

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767696

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767808

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767827

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767854

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767859

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767446) |
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765230258/job/94659767706) |


## [Run #31765141322](https://github.com/sgl-project/sglang/actions/runs/31765141322)
- **分支**: `kurt/moe-runner-extension-points`
- **总耗时**: 306.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31765141322

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 17.9min | 代码错误 | NPU测试中test_npu_autoround_moe.py执行失败，退出码为1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384494) |
| base-b-test-8-npu-a3 / run (0) | 6.2min | 代码错误 | NPU专家并行测试用例失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384523) |
| base-b-test-4-npu-a3 / run (1) | 6.8min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384532) |
| base-b-test-2-npu-a3 / run (0) | 8.3min | 代码错误 | NPU专家并行测试用例失败，导致作业整体退出 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384534) |
| base-b-test-4-npu-a3 / run (0) | 5.8min | 环境问题 | NPU测试用例test_npu_hicache_mla.py执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384575) |
| base-b-test-16-npu-a3 / run (0) | 50.4min | 代码错误 | NPU测试test_npu_deepep.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384579) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.4min | 精度回归 | GLM4-7B GSM8K 精度测试失败，0/3 用例通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384777) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.7min | 精度回归 | NPU精度测试glm5_top64_pruned_bf16_8p_gsm8k失败，0/1通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384863) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 15.6min | 精度回归 | qwen3_vl_30b_a3b_bf16_2p_gsm8k测试失败，精度未达标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384890) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94702935667) |

- **base-b-test-1-npu-a3 / run (0)**: 测试文件test_npu_autoround_moe.py在运行过程中失败（exit code 1），导致整个CI作业终止。该测试属于量化功能测试，可能涉及MoE模型的autoround量化实现存在代码问题或环境依赖缺失。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384494

- **base-b-test-8-npu-a3 / run (0)**: 测试文件test_npu_eplb_min_rebalancing_utilization_threshold.py执行失败，耗时175秒，0/1测试通过。可能是EPLB最小再平衡利用率阈值相关逻辑或配置存在问题，需检查该测试用例的具体断言和实现。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384523

- **base-b-test-4-npu-a3 / run (1)**: 测试文件test/registered/npu/basic_function/dllm/test_npu_llada2_mini.py在运行174秒后失败，退出码为1，导致整个作业终止。具体失败原因需查看该测试的详细输出，可能是功能实现或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384532

- **base-b-test-2-npu-a3 / run (0)**: 测试文件test_npu_expert_distribution_recorder_mode.py执行失败（exit code 1），6个测试全部未通过，最终导致作业以255退出。具体失败原因需查看该测试的详细输出，但可确定是测试代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384534

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test/registered/npu/basic_function/HiCache/test_npu_hicache_mla.py在运行113秒后失败，退出码为1，导致整个作业失败。日志未显示具体错误原因，可能是NPU环境或测试用例本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384575

- **base-b-test-16-npu-a3 / run (0)**: 测试test/registered/npu/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep.py运行344秒后失败（exit code 1），其余两个测试通过，表明该测试用例本身存在代码或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384579

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 test_npu_glm4_7_flash_1p_gsm8k.py 返回退出码 1，耗时 173.93 秒，3 个测试全部失败，属于精度回归问题，可能由模型权重、推理配置或代码改动引起。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384777

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py返回退出码1，耗时355秒，属于精度测试未通过，可能因模型精度或数据问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384863

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 测试套件中qwen3_vl_30b_a3b_bf16_2p_gsm8k.py返回退出码1，而qwen3_vl_8b测试通过，表明30b模型在GSM8K任务上精度回归，可能因模型权重或推理配置变化导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384890

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1076秒后退出码1，属于性能测试未通过，可能因吞吐或延迟未达预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94702935667

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384464) |
| base-a-test-1-npu-a2 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384615) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31765141322/job/94659384829) |


## [Run #31764988709](https://github.com/sgl-project/sglang/actions/runs/31764988709)
- **分支**: `main`
- **总耗时**: 20.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31764988709

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.6min | 环境问题 | 作业因环境问题失败，未找到失败产物文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658911991) |
| base-a-test-1-npu-a2 / run (0) | 3.8min | 环境问题 | 自定义容器执行失败，下载triton-ascend时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912035) |
| base-b-test-4-npu-a3 / run (1) | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912094) |
| base-b-test-4-npu-a3 / run (0) | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912097) |
| base-b-test-2-npu-a3 / run (0) | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912111) |
| base-b-test-8-npu-a3 / run (0) | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912212) |
| base-b-test-1-npu-a3 / run (0) | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912221) |
| base-b-test-16-npu-a3 / run (0) | 19.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912224) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912504) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912532) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912615) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912654) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在运行后上传diffusion-failures/目录时提示无文件，说明测试未产生失败记录，可能因环境配置或依赖问题导致测试未正常执行，而非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658911991

- **base-a-test-1-npu-a2 / run (0)**: 作业在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中，自定义容器实现执行失败，导致作业终止。可能是网络或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912035

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912094

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912097

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912111

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912212

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912221

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912224

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912504

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912532

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912615

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764988709/job/94658912654


## [Run #31764978526](https://github.com/sgl-project/sglang/actions/runs/31764978526)
- **分支**: `codex/component-residency-policy`
- **总耗时**: 40.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31764978526

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 38.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764978526/job/94658949740) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅有GitHub Actions环境准备、Node版本警告及上传失败产物（无文件）等常规信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764978526/job/94658949740


## [Run #31764589472](https://github.com/sgl-project/sglang/actions/runs/31764589472)
- **分支**: `new_epd`
- **总耗时**: 323.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31764589472

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 40.8min | 性能回归 | 性能测试中qwen3_235b_w8a8用例失败，疑似性能未达标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94694934335) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94707993565) |

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示两个性能测试通过，但qwen3_235b_a22b用例退出码1，耗时1251秒，远超其他用例，可能因性能指标未达阈值或运行异常导致失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94694934335

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示该作业在启动前因同一PR中另一个作业（base-c-test-perf-16-npu-a3）失败而被fast-fail跳过，属于依赖失败导致的级联取消，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94707993565

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657763685) |
| multimodal-gen-test-1-npu-a3 | 25.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657763716) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657763777) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657763786) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657763790) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657763823) |
| base-b-test-4-npu-a3 / run (0) | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657763845) |
| base-b-test-16-npu-a3 / run (0) | 50.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657763905) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657764068) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657764110) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657764150) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 111.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94657764258) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94691368056) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764589472/job/94695253361) |


## [Run #31764399793](https://github.com/sgl-project/sglang/actions/runs/31764399793)
- **分支**: `fuse-swiglu-moe-up-gemm-epilogue`
- **总耗时**: 127.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31764399793

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 126.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236172) |
| base-b-test-8-npu-a3 / run (0) | 126.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236202) |
| base-b-test-1-npu-a3 / run (0) | 126.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236223) |
| base-b-test-16-npu-a3 / run (0) | 126.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236247) |
| base-b-test-4-npu-a3 / run (1) | 126.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236266) |
| base-b-test-4-npu-a3 / run (0) | 126.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236278) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 126.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236354) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 126.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236381) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 126.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236397) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 126.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236417) |

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236172

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236202

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236223

- **base-b-test-16-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，说明CI流程尝试访问的存储对象缺失，可能是上传失败、路径错误或资源被清理，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236247

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236266

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236278

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236354

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236381

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储对象缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236397

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236417

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236116) |
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31764399793/job/94657236208) |


## [Run #31763861002](https://github.com/sgl-project/sglang/actions/runs/31763861002)
- **分支**: `xinyuan/cpu-ci-prune-static-overhead`
- **总耗时**: 22.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31763861002

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 21.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730447) |
| multimodal-gen-test-1-npu-a3 | 8.4min | 其他 | 日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730450) |
| base-b-test-8-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730481) |
| base-a-test-1-npu-a2 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730489) |
| base-b-test-2-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730502) |
| base-b-test-4-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730510) |
| base-b-test-1-npu-a3 / run (0) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730516) |
| base-b-test-4-npu-a3 / run (1) | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730591) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730639) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730675) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730693) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 21.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730699) |

- **base-b-test-16-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，说明CI流程尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730447

- **multimodal-gen-test-1-npu-a3**: 作业在运行后上传diffusion-failures目录时提示无文件，说明测试可能通过或失败未产生产物。日志中间部分被省略，无法定位具体错误，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730450

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730481

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730489

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730502

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730510

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730516

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730591

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730639

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730675

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730693

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）未上传或已被删除，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763861002/job/94655730699


## [Run #31763732823](https://github.com/sgl-project/sglang/actions/runs/31763732823)
- **分支**: `cc-fixes-rebased`
- **总耗时**: 293.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31763732823

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.8min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94686710897) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1.0min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94693428573) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94697331225) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94701503243) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1093秒后失败，0/1通过。该测试为性能基准测试，失败可能因性能未达阈值或环境问题，需查看详细日志确认具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94686710897

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业触发了fast-fail，导致本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94693428573

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94697331225

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94701503243

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655290747) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655290801) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655290804) |
| base-b-test-4-npu-a3 / run (0) | 31.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655290820) |
| base-b-test-16-npu-a3 / run (0) | 51.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655290830) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655290835) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655290838) |
| base-b-test-8-npu-a3 / run (0) | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655290861) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655291099) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 49.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655291105) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655291121) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763732823/job/94655291127) |


## [Run #31763699808](https://github.com/sgl-project/sglang/actions/runs/31763699808)
- **分支**: `feat/graceful-shutdown`
- **总耗时**: 328.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31763699808

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 131.6min | 精度回归 | qwen3_5_9b_bf16_1p_gsm8k 测试失败，导致作业整体退出码非零。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194896) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94685019668) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94688871181) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94691627699) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 在 NPU 精度测试中，glm4_7_flash 和 moonlight_16b 均通过，但 qwen3_5_9b_bf16_1p_gsm8k 测试退出码为 1，表明该模型精度未达预期，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194896

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1114秒后退出码为1，属于性能测试未通过，可能因吞吐或延迟未达预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94685019668

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，作为根因作业触发了fast-fail，导致本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94688871181

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因失败作业为base-c-test-perf-8-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94691627699

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194612) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194667) |
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194675) |
| multimodal-gen-test-1-npu-a3 | 27.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194696) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194700) |
| base-b-test-1-npu-a3 / run (0) | 25.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194711) |
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194713) |
| base-b-test-16-npu-a3 / run (0) | 50.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194783) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194814) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194841) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763699808/job/94655194966) |


## [Run #31763302028](https://github.com/sgl-project/sglang/actions/runs/31763302028)
- **分支**: `cctry/http2-max-concurrent-streams`
- **总耗时**: 291.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31763302028

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 69.9min | 精度回归 | NPU精度测试失败，qwen3_5_9b_bf16_1p_gsm8k测试用例返回退出码1，0/3测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654031099) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | 性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94683885606) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 25.4min | 超时 | 性能测试用例执行超时导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94690499093) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py执行3944秒后失败，退出码1，所有3个测试均未通过，属于精度回归问题，可能由模型输出与预期不符导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654031099

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1121秒后失败，0/1通过，属于性能测试未达标，可能因吞吐或延迟未满足50ms要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94683885606

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py运行1293秒后失败，超过预估时间3600秒，最终0/4测试通过，作业以退出码255终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94690499093

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654030652) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654030747) |
| base-b-test-4-npu-a3 / run (0) | 30.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654030756) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654030760) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654030788) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654030796) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654030841) |
| base-b-test-16-npu-a3 / run (0) | 52.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654030883) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654031064) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 37.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654031087) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94654031132) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31763302028/job/94689728164) |


## [Run #31763025422](https://github.com/sgl-project/sglang/actions/runs/31763025422)
- **分支**: `codex/component-residency-policy`
- **总耗时**: 38.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31763025422

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.7min | 其他 | 日志不完整，未显示测试失败的具体原因，仅包含环境警告和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31763025422/job/94653214130) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅有Node.js版本弃用警告和上传diffusion-failures产物时未找到文件的提示，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31763025422/job/94653214130


## [Run #31762567486](https://github.com/sgl-project/sglang/actions/runs/31762567486)
- **分支**: `fix/issue-31766-fd-exhaustion`
- **总耗时**: 322.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31762567486

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94688753588) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 26.6min | 性能回归 | NPU性能测试未达标，qwen3_235b_w8a8_8p_in3k5_out1k5_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94692850437) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94694380598) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94707681369) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1043秒后退出码1，该测试为性能测试，预期耗时3600秒，实际提前失败，可能因性能指标未达阈值或运行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94688753588

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试脚本test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py执行失败，0/4测试通过，耗时1362秒，未达到预期性能指标，可能因模型推理速度不满足50ms延迟要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94692850437

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，由于快速失败策略，本作业未实际运行即被终止，属于上游失败导致的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94694380598

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在健康检查阶段因其他根因作业（base-c-test-perf-8/16-npu-a3）失败而触发快速失败机制，本作业本身未执行测试，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94707681369

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487104) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487109) |
| base-b-test-4-npu-a3 / run (0) | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487127) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487136) |
| multimodal-gen-test-1-npu-a3 | 25.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487142) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487163) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 115.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487197) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487199) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487204) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487206) |
| base-b-test-16-npu-a3 / run (0) | 47.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487212) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762567486/job/94657487265) |


## [Run #31762448440](https://github.com/sgl-project/sglang/actions/runs/31762448440)
- **分支**: `fix/qwen35-hicache-mtp-draft-depth`
- **总耗时**: 253.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31762448440

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 56.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542881) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 28.3min | 性能回归 | NPU性能测试中kimi_k2_6用例失败，耗时898秒未达标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94684852773) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 16.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94687582804) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中服务正常响应，但随后出现“Executing the custom container implementation failed”错误，属于runner容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542881

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件4个用例中1个失败，kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms用例返回退出码1，耗时898秒，未满足20ms性能目标，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94684852773

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94687582804

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542684) |
| multimodal-gen-test-1-npu-a3 | 31.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542709) |
| base-b-test-8-npu-a3 / run (0) | 9.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542729) |
| base-b-test-16-npu-a3 / run (0) | 46.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542733) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542740) |
| base-b-test-4-npu-a3 / run (0) | 29.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542763) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542764) |
| base-a-test-1-npu-a2 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542776) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651542944) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651543022) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94651543043) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762448440/job/94680720226) |


## [Run #31762381721](https://github.com/sgl-project/sglang/actions/runs/31762381721)
- **分支**: `main`
- **总耗时**: 27.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31762381721

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 26.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387488) |
| base-b-test-16-npu-a3 / run (0) | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387496) |
| base-b-test-8-npu-a3 / run (0) | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387505) |
| base-b-test-4-npu-a3 / run (1) | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387636) |
| base-b-test-4-npu-a3 / run (0) | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387637) |
| base-b-test-1-npu-a3 / run (0) | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387644) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387819) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387835) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387846) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387955) |

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或依赖文件未生成，属于环境或基础设施问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387488

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387496

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387505

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387636

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387637

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387644

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387819

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置变更导致，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387835

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387846

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387955

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762381721/job/94651387635) |


## [Run #31762274283](https://github.com/sgl-project/sglang/actions/runs/31762274283)
- **分支**: `fix/qwen35-fused-qk-rmsnorm-zerogrid-31350`
- **总耗时**: 260.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31762274283

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.9min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94674049633) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94680061279) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他作业（8-NPU）已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94680358288) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94687980534) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1107秒后失败，该测试为性能测试，预期耗时3600秒，实际提前退出且未通过，属于性能未达标或执行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94674049633

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，属于根因失败，因此本作业被快速失败（fast-fail）跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94680061279

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示本作业未实际运行，而是在健康检查阶段因检测到base-c-test-perf-8-npu-a3作业失败而触发fast-fail机制，主动跳过执行。属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94680358288

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业被级联跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94687980534

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014556) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014657) |
| base-b-test-4-npu-a3 / run (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014678) |
| base-b-test-2-npu-a3 / run (0) | 18.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014698) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014705) |
| base-b-test-1-npu-a3 / run (0) | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014746) |
| base-a-test-1-npu-a2 / run (0) | 10.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014781) |
| base-b-test-16-npu-a3 / run (0) | 51.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014808) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014930) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 105.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014950) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014965) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762274283/job/94651014990) |


## [Run #31762265214](https://github.com/sgl-project/sglang/actions/runs/31762265214)
- **分支**: `codex/sglang-phase-a-admission-rebased-20260810`
- **总耗时**: 14.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31762265214

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995566) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995608) |
| base-b-test-2-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995637) |
| base-b-test-8-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995645) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995664) |
| base-b-test-1-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995686) |
| base-b-test-4-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995730) |
| base-b-test-16-npu-a3 / run (0) | 14.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995782) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995805) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995811) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995825) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995863) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995566

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动自定义容器实现时失败，错误提示需联系自托管runner管理员，属于NPU CI基础设施或容器环境配置问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995608

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995637

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995645

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995664

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995686

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995730

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995782

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995805

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995811

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995825

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762265214/job/94650995863


## [Run #31762212381](https://github.com/sgl-project/sglang/actions/runs/31762212381)
- **分支**: `unidy2002/weight-cache-static-dp-ep`
- **总耗时**: 63.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31762212381

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅包含环境警告和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827461) |
| base-b-test-4-npu-a3 / run (1) | 63.2min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827509) |
| base-b-test-4-npu-a3 / run (0) | 63.2min | 环境问题 | CI 日志中引用的 Azure Blob 存储对象不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827529) |
| base-b-test-2-npu-a3 / run (0) | 63.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827571) |
| base-b-test-1-npu-a3 / run (0) | 63.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827579) |
| base-b-test-16-npu-a3 / run (0) | 63.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827611) |
| base-b-test-8-npu-a3 / run (0) | 63.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827619) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 63.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827624) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 63.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827629) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 63.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827639) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 63.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827664) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。可见部分仅为Node.js 20弃用警告、上传diffusion-failures产物时未找到文件等非致命信息，实际测试失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827461

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827509

- **base-b-test-4-npu-a3 / run (0)**: 作业运行时尝试下载某个 blob 文件，但该文件已被删除或路径错误，返回 BlobNotFound 错误。这属于外部存储资源缺失或配置不一致，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827529

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827571

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827579

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827611

- **base-b-test-8-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，说明CI流程尝试访问的Azure Blob存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827619

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827624

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827629

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827639

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及文件可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827664

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762212381/job/94650827487) |


## [Run #31762170839](https://github.com/sgl-project/sglang/actions/runs/31762170839)
- **分支**: `opt/kimi-k2-mxfp4-fp8-bmm-direct-write`
- **总耗时**: 230.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31762170839

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 21.0min | 代码错误 | NPU DP注意力测试失败，1/5用例通过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715699) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94670693740) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业被健康检查快速失败机制跳过，因其他相关作业已失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94675405890) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94676563371) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94683357553) |

- **base-b-test-4-npu-a3 / run (0)**: test_npu_dp_attention.py测试返回退出码1，耗时847秒，导致整个作业失败。其余4个测试中仅1个通过，表明DP注意力功能存在代码或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715699

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，耗时1134秒，未通过性能指标要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94670693740

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业未实际运行，因健康检查检测到同PR中base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，触发了fast-fail跳过逻辑，属于依赖作业失败导致的级联取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94675405890

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示该作业在启动前被PR健康检查拦截，因其他根因作业（base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3）失败而触发fast-fail，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94676563371

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业（base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3），本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94683357553

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715439) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715675) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715741) |
| base-b-test-4-npu-a3 / run (1) | 17.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715742) |
| base-b-test-8-npu-a3 / run (0) | 8.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715799) |
| base-b-test-1-npu-a3 / run (0) | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715801) |
| base-b-test-16-npu-a3 / run (0) | 53.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650715831) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650716016) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650716045) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650716051) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762170839/job/94650716054) |


## [Run #31762168366](https://github.com/sgl-project/sglang/actions/runs/31762168366)
- **分支**: `opt/kimi-k2-mxfp4-fuse-pertoken-quant`
- **总耗时**: 226.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31762168366

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 39.1min | 其他 | 作业被健康检查快速失败机制跳过，因另一作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689557) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.4min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94666541721) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94674390355) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94683277017) |

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业本身未运行测试，因健康检查发现根因失败作业base-c-test-perf-8-npu-a3，触发fast-fail跳过本作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689557

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1136秒后失败，0/1通过。该测试为性能测试，失败原因可能是性能未达到预设阈值（如50ms延迟要求），需检查具体性能指标是否达标。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94666541721

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到根因作业 base-c-test-perf-8-npu-a3 失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94674390355

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被快速跳过，并非自身代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94683277017

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689145) |
| multimodal-gen-test-1-npu-a3 | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689169) |
| base-b-test-2-npu-a3 / run (0) | 18.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689189) |
| base-a-test-1-npu-a2 / run (0) | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689203) |
| base-b-test-16-npu-a3 / run (0) | 55.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689230) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689235) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689274) |
| base-b-test-4-npu-a3 / run (0) | 28.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689285) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689380) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689450) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31762168366/job/94650689471) |


## [Run #31761743098](https://github.com/sgl-project/sglang/actions/runs/31761743098)
- **分支**: `real_max_prefill_size`
- **总耗时**: 191.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31761743098

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.7min | 性能回归 | NPU性能测试未达预期，测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94663507607) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.8min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94668426616) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94668641678) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（perf-8）已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94676963760) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1079秒后失败，该用例为性能测试，可能因推理性能未达到设定阈值（如50ms）导致失败，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94663507607

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 健康检查发现base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，实际未执行测试，属于上游失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94668426616

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（4-npu）被快速失败跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94668641678

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（perf-2）被级联过滤并快速失败，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94676963760

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446275) |
| base-a-test-1-npu-a2 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446315) |
| base-b-test-4-npu-a3 / run (1) | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446330) |
| base-b-test-2-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446337) |
| base-b-test-4-npu-a3 / run (0) | 32.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446370) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446392) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446401) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446488) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446507) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446521) |
| base-b-test-16-npu-a3 / run (0) | 59.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446525) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761743098/job/94649446598) |


## [Run #31761589073](https://github.com/sgl-project/sglang/actions/runs/31761589073)
- **分支**: `codex/extensible-serve-backends`
- **总耗时**: 192.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31761589073

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.6min | 性能回归 | NPU性能测试未通过，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94663401012) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现同批次8卡性能作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94667973832) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94668052497) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-NPU）已失败而跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94675435028) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1121秒后失败，返回码1，0/1测试通过，属于性能测试未达标。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94663401012

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，导致本4卡作业在启动前被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94667973832

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94668052497

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业未实际运行测试，在健康检查阶段因检测到根因作业 base-c-test-perf-8-npu-a3 失败而触发 fast-fail 跳过，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94675435028

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 18.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971391) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971393) |
| base-b-test-16-npu-a3 / run (0) | 50.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971405) |
| base-b-test-4-npu-a3 / run (0) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971480) |
| base-a-test-1-npu-a2 / run (0) | 8.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971485) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971490) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971494) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971764) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971769) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971841) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761589073/job/94648971897) |


## [Run #31761245774](https://github.com/sgl-project/sglang/actions/runs/31761245774)
- **分支**: `streaming_session`
- **总耗时**: 203.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31761245774

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 4.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950629) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.6min | 性能回归 | NPU性能测试未通过，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94658559583) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94664612063) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94666056679) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94677515548) |

- **base-b-test-16-npu-a3 / run (0)**: 健康检查检测到根因失败作业base-c-test-perf-8-npu-a3，根据快速失败策略，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950629

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py返回退出码1，耗时约1170秒，未达到性能要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94658559583

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94664612063

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94666056679

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3作业失败，本作业被标记为级联失败并跳过，实际未执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94677515548

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950463) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950540) |
| base-b-test-4-npu-a3 / run (0) | 28.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950547) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950591) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950605) |
| base-b-test-2-npu-a3 / run (0) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950660) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950789) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950812) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950829) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950848) |
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31761245774/job/94647950875) |


## [Run #31760777644](https://github.com/sgl-project/sglang/actions/runs/31760777644)
- **分支**: `pllimax/output-log-dir-structure`
- **总耗时**: 193.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31760777644

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 24.2min | 性能回归 | NPU性能测试未达预期，minimax_m2_5测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94658430494) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因健康检查快速失败机制被跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94663652209) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他作业失败导致本作业被快速跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94664989953) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8卡性能测试）已失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94674331637) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1135秒后失败，退出码1，0/1通过，属于性能指标未达标导致的回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94658430494

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示该作业未实际运行，而是因同一PR中另一个作业base-c-test-perf-8-npu-a3失败触发了fast-fail跳过。根因是其他作业失败，本作业被连带取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94663652209

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段因健康检查检测到base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业被跳过未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94664989953

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示根因失败作业为base-c-test-perf-8-npu-a3，本作业（2卡）因健康检查机制被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94674331637

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515477) |
| base-a-test-1-npu-a2 / run (0) | 10.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515529) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515548) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515584) |
| base-b-test-8-npu-a3 / run (0) | 10.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515617) |
| base-b-test-16-npu-a3 / run (0) | 51.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515686) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515749) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515876) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515926) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515934) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 88.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760777644/job/94646515944) |


## [Run #31760509867](https://github.com/sgl-project/sglang/actions/runs/31760509867)
- **分支**: `fix/qwen35-gdn-mis`
- **总耗时**: 55.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31760509867

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 54.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708383) |
| base-b-test-16-npu-a3 / run (0) | 54.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708409) |
| base-b-test-1-npu-a3 / run (0) | 54.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708410) |
| base-b-test-4-npu-a3 / run (1) | 54.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708471) |
| base-b-test-4-npu-a3 / run (0) | 54.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708482) |
| base-b-test-2-npu-a3 / run (0) | 54.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708601) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 54.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708644) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 54.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708686) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 54.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708698) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 54.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708699) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708383

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708409

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存、工件或依赖文件未正确上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708410

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708471

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708482

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708601

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708644

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708686

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708698

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708699

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 32.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708373) |
| base-a-test-1-npu-a2 / run (0) | 9.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31760509867/job/94645708470) |


## [Run #31759434705](https://github.com/sgl-project/sglang/actions/runs/31759434705)
- **分支**: `jit-content-addressed-cache`
- **总耗时**: 277.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31759434705

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 44.1min | 性能回归 | kimi_k2_6性能测试未通过，耗时1460秒远超预期20ms目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94661987217) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 51.1min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94674021808) |

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms测试失败，退出码1，耗时1460秒，远超20ms性能目标，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94661987217

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示测试运行正常，但执行到一半时出现“Executing the custom container implementation failed”错误，可能是容器环境或runner问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94674021808

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478364) |
| base-b-test-4-npu-a3 / run (0) | 31.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478398) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478413) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478420) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478483) |
| base-b-test-1-npu-a3 / run (0) | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478554) |
| base-b-test-16-npu-a3 / run (0) | 58.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478567) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478647) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 107.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478699) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478730) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478780) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94642478798) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94657034285) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759434705/job/94663888312) |


## [Run #31759311404](https://github.com/sgl-project/sglang/actions/runs/31759311404)
- **分支**: `feature/optimize-paged-mqa-metadata`
- **总耗时**: 58.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31759311404

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113534) |
| base-b-test-16-npu-a3 / run (0) | 58.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113568) |
| base-b-test-1-npu-a3 / run (0) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113601) |
| base-b-test-4-npu-a3 / run (1) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113621) |
| base-b-test-4-npu-a3 / run (0) | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113659) |
| base-b-test-2-npu-a3 / run (0) | 58.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113707) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113908) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113920) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113929) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 58.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642114006) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113534

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113568

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113601

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113621

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113659

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载日志文件，但返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113707

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113908

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113920

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113929

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642114006

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759311404/job/94642113650) |


## [Run #31759197901](https://github.com/sgl-project/sglang/actions/runs/31759197901)
- **分支**: `main`
- **总耗时**: 60.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31759197901

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 23.8min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750671) |
| base-b-test-4-npu-a3 / run (1) | 22.1min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750728) |
| base-b-test-16-npu-a3 / run (0) | 21.1min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750743) |
| base-b-test-8-npu-a3 / run (0) | 60.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750745) |
| base-b-test-2-npu-a3 / run (0) | 23.7min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750823) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 代码错误 | NPU HiCache MLA 测试失败，返回退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750865) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 22.4min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641751082) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.6min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641751141) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 60.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641751144) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94650002324) |

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重后，自定义容器实现执行失败，提示联系自托管runner管理员。可能是容器环境或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750671

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行正常（吞吐量正常），但在02:00:27时出现"Executing the custom container implementation failed"错误，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750728

- **base-b-test-16-npu-a3 / run (0)**: 日志显示容器启动后Watchdog TokenizerManager等初始化正常，但随后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于NPU容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750743

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败、过期或被误删，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750745

- **base-b-test-2-npu-a3 / run (0)**: 日志显示测试运行正常（HTTP 200），但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750823

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 执行失败，耗时281秒，0/5测试通过，具体错误未在日志中显示，可能涉及功能或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750865

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但中途出现“Executing the custom container implementation failed”错误，提示联系自托管runner管理员，属于容器环境异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641751082

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示测试运行中突然报错'Executing the custom container implementation failed'，提示联系runner管理员，属于NPU自托管runner环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641751141

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641751144

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94650002324

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750575) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641750828) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31759197901/job/94641751112) |


## [Run #31758802795](https://github.com/sgl-project/sglang/actions/runs/31758802795)
- **分支**: `k3_dcp_1n`
- **总耗时**: 151.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31758802795

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 28.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525297) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | NPU性能测试未通过，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94644249298) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94647944180) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败被快速失败跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94650624308) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94662358696) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525297

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1115秒后失败，返回退出码1，属于性能测试未达标或执行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94644249298

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，触发了fast-fail机制，本作业未实际执行测试即被跳过，属于上游失败导致的连锁取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94647944180

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 健康检查发现根因作业multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3失败，本作业被级联跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94650624308

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3和base-c-test-perf-8-npu-a3，本作业因级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94662358696

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525303) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525306) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525342) |
| base-b-test-1-npu-a3 / run (0) | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525345) |
| base-b-test-2-npu-a3 / run (0) | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525375) |
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525433) |
| base-b-test-16-npu-a3 / run (0) | 53.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525473) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525596) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525634) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 131.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525658) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758802795/job/94640525732) |


## [Run #31758721636](https://github.com/sgl-project/sglang/actions/runs/31758721636)
- **分支**: `refactor-mxfp4-sm100-trtllm-moerunner`
- **总耗时**: 210.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31758721636

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 25.1min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94643529636) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 34.1min | 性能回归 | NPU性能测试中kimi_k2_6用例失败，未达性能目标。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94647075386) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1123秒后退出码为1，属于性能测试未通过，可能因模型推理速度未达到预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94643529636

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件1/4通过，kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms用例退出码1，耗时1455秒，可能因性能不达标或运行错误导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94647075386

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 40.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324481) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324595) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324656) |
| base-b-test-16-npu-a3 / run (0) | 53.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324687) |
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324697) |
| base-b-test-4-npu-a3 / run (0) | 30.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324723) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324780) |
| base-b-test-4-npu-a3 / run (1) | 13.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324838) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324918) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324936) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324946) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94640324952) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94647552121) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758721636/job/94658955211) |


## [Run #31758298364](https://github.com/sgl-project/sglang/actions/runs/31758298364)
- **分支**: `online-nvfp4-to-mxfp4-convert`
- **总耗时**: 193.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31758298364

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639949774) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 48.2min | 性能回归 | Kimi K2 6 性能测试未通过，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94643259997) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1127秒后退出码为1，性能指标未达标，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639949774

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 性能测试套件中 kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms.py 测试失败（exit code 1），而其他两个测试通过，表明该模型性能未达预期标准。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94643259997

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639007809) |
| base-b-test-1-npu-a3 / run (0) | 22.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639007924) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639007937) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008010) |
| base-b-test-16-npu-a3 / run (0) | 53.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008026) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008037) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008067) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008078) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008225) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008268) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008295) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94639008358) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94644882210) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758298364/job/94652732517) |


## [Run #31758217869](https://github.com/sgl-project/sglang/actions/runs/31758217869)
- **分支**: `fix/has-hf-quant-config-local-dirs`
- **总耗时**: 172.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31758217869

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.2min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94653650932) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因健康检查发现其他根因作业失败而被快速跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94658758824) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94659063541) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 环境问题 | 健康检查发现其他作业失败，导致本作业被快速失败跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94665131973) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1179秒后失败，该测试为性能测试，预期耗时3600秒，但实际未通过，可能因性能指标未达到要求（如延迟或吞吐量不达标）而判定失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94653650932

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到另一个作业base-c-test-perf-8-npu-a3失败，触发了fast-fail机制，导致本作业未实际运行即被跳过，属于CI流程中的级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94658758824

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被过滤，最终因根因作业失败而触发fast-fail机制，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94659063541

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示base-c-test-perf-8-npu-a3作业失败，健康检查将其识别为根因，本作业作为级联失败被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94665131973

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864203) |
| base-a-test-1-npu-a2 / run (0) | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864248) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864276) |
| base-b-test-2-npu-a3 / run (0) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864291) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864298) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864313) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864389) |
| base-b-test-16-npu-a3 / run (0) | 51.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864443) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864644) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864702) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864744) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31758217869/job/94641864873) |


## [Run #31757764626](https://github.com/sgl-project/sglang/actions/runs/31757764626)
- **分支**: `main`
- **总耗时**: 7.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31757764626

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.0min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363752) |
| base-b-test-2-npu-a3 / run (0) | 5.9min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363775) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363781) |
| base-b-test-4-npu-a3 / run (1) | 6.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363783) |
| base-b-test-1-npu-a3 / run (0) | 5.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363789) |
| base-b-test-4-npu-a3 / run (0) | 6.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363801) |
| base-b-test-16-npu-a3 / run (0) | 6.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363846) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363885) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.5min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637364021) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.9min | 环境问题 | 自定义容器执行失败，模型权重加载中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637364033) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94638263565) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行测试后上传diffusion-failures目录时提示无文件，但日志中间部分被省略，无法定位具体失败点。可能是测试未产生失败文件或测试本身未执行成功。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363752

- **base-b-test-2-npu-a3 / run (0)**: 日志显示服务启动成功并完成一次生成请求，但随后报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363775

- **base-b-test-8-npu-a3 / run (0)**: 作业在运行约6分钟后，日志显示"Executing the custom container implementation failed"，提示联系自托管runner管理员，表明NPU容器环境在执行过程中出现故障，导致测试提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363781

- **base-b-test-4-npu-a3 / run (1)**: 日志显示测试运行中容器突然崩溃，报错'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363783

- **base-b-test-1-npu-a3 / run (0)**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363789

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试进行到batch size捕获阶段时，自定义容器实现执行失败，提示联系self-hosted runner管理员。可能是NPU资源或容器环境问题导致作业中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363801

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363846

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363885

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在加载模型分片过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637364021

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在加载Qwen3-VL模型权重（加载至75%）时，自定义容器实现执行失败，导致测试提前终止。日志显示为runner容器环境问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637364033

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据文件在 Azure Blob 存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94638263565

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637363731) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31757764626/job/94637364000) |


---
*Auto-generated by npu_pr_monitor.py*