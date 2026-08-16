# NPU CI 执行监控
**生成时间**: 2026-08-16 08:16 UTC
**分析 Run 数**: 26

---

## 📊 本次执行总结

- **成功 Job 数**: 93
- **失败 Run 数**: 22
- **成功 Job 平均耗时**: 30.7min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 265.8min | #31910955003 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95079613749) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 128.4min | #31910955003 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828294) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 123.3min | #31486917906 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512501) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 120.6min | #31915187512 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903964) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.1min | #31487148563 | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118217) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.2min | #31485916694 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248793) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.7min | #31914132334 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474489) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.6min | #31915127701 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783169) |
| base-b-test-16-npu-a3 / run (0) | 69.0min | #31915187512 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903798) |
| base-b-test-16-npu-a3 / run (0) | 55.3min | #31914132334 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474387) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31910955003<br>[#24911 Profiling Enhancements [2/3]: detailed execution step annotations](https://github.com/sgl-project/sglang/pull/24911) | `feat/roofline_annotations` | 301.4min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31910955003) |
| #31486917906<br>[#33997 Bump FlashInfer to 0.6.17 and remove Kimi K3 workarounds](https://github.com/sgl-project/sglang/pull/33997) | `mmangkad/flashinfer-0.6.17rc1-kimi-k3` | 182.1min | 5 | base-b-test-16-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31486917906) |
| #31487148563<br>[#32882 [Bugfix] Accept int64 top-k IDs in FlashInfer routed MoE packer](https://github.com/sgl-project/sglang/pull/32882) | `fix-pack-topk-int64` | 177.9min | 5 | base-b-test-16-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31487148563) |
| #31485916694<br>[#32313 [Feature] Optimize TP LMHead with All-to-All](https://github.com/sgl-project/sglang/pull/32313) | `lm-head-opt` | 144.6min | 5 | base-b-test-16-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31485916694) |
| #31915187512<br>[#34763 [Spec] Support mamba-radix-cache-strategy extra_buffer_lazy with DFLASH](https://github.com/sgl-project/sglang/pull/34763) | `main` | 141.0min | 4 | base-b-test-4-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31915187512) |
| #31915127701<br>[#34982 [misc] Rename shared-read boundary to shared-read ends and fix wrapper delegation](https://github.com/sgl-project/sglang/pull/34982) | `lsyin/shared-read-default-pre-replay` | 130.8min | 3 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31915127701) |
| #31914132334<br>[#30575 [AMD] Enable Fast Triton Sparse MLA backend](https://github.com/sgl-project/sglang/pull/30575) | `feat/triton-sparse-mla` | 91.4min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31914132334) |
| #31486012963 | `diffusion-ideogram-rope-silu-fusion` | 67.5min | 10 | base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31486012963) |
| #31485463912<br>[#33676 [NPU] Support DeepSeek-V4 DSpark and refactor DSV4 cache management](https://github.com/sgl-project/sglang/pull/33676) | `main_8.5` | 59.5min | 11 | base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31485463912) |
| #31491110682<br>[#34314 [diffusion] Ideogram-4: fuse Qwen3-style RoPE and SwiGLU silu-mul (denoise -5.1% H100 / -4.7% H200, bit-exact)](https://github.com/sgl-project/sglang/pull/34314) | `diffusion-ideogram-rope-silu-fusion` | 51.9min | 11 | base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31491110682) |
| #31491657646<br>[#32162 [HiSparse] Support hisparse multi-step swap io kernel](https://github.com/sgl-project/sglang/pull/32162) | `hisparse_mtp_kernel` | 48.2min | 11 | base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31491657646) |
| #31484207064<br>[#34348 [Diffusion] Restore ERNIE norm fusion on SM120](https://github.com/sgl-project/sglang/pull/34348) | `agent/optimize-sm120-ernie-norm-fusion` | 39.6min | 9 | base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31484207064) |
| #31490838473 | `codex/cpu-offload-components-clean` | 34.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31490838473) |
| #31489956661<br>[#29070 [DSV4] perf: Enable alt stream during BCG prefill](https://github.com/sgl-project/sglang/pull/29070) | `main` | 30.5min | 11 | base-b-test-16-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31489956661) |
| #31493433076<br>[#34307 [GDN] Fused qkvzba split for non-pow2 v-head ratios, default FlashInfer GDN prefill on SM90, and an opt-in Hopper bf16 GEMV backend](https://github.com/sgl-project/sglang/pull/34307) | `gdn-ratio3-flashinfer-prefill-sm90` | 23.3min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31493433076) |
| #31493026763<br>[#34341 [npu] [bugfix] Fix HiCache MHA backup for NPU](https://github.com/sgl-project/sglang/pull/34341) | `pd-fix-2` | 21.5min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31493026763) |
| #31492376732<br>[#34257 [JIT Kernel] Migrate per-token FP8 quantization from AOT to JIT](https://github.com/sgl-project/sglang/pull/34257) | `main` | 20.4min | 11 | base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31492376732) |
| #31495329896<br>[#34307 [GDN] Fused qkvzba split for non-pow2 v-head ratios, default FlashInfer GDN prefill on SM90, and an opt-in Hopper bf16 GEMV backend](https://github.com/sgl-project/sglang/pull/34307) | `gdn-ratio3-flashinfer-prefill-sm90` | 11.2min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31495329896) |
| #31494765760<br>[#34341 [npu] [bugfix] Fix HiCache MHA backup for NPU](https://github.com/sgl-project/sglang/pull/34341) | `main` | 11.1min | 12 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31494765760) |
| #31494226228<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 9.8min | 12 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31494226228) |
| #31492513829<br>[#33829 [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding](https://github.com/sgl-project/sglang/pull/33829) | `dots-note-for-sglang` | 9.7min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31492513829) |
| #31483724651 | `agent/optimize-sm120-ernie-norm-fusion` | 7.0min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31483724651) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 21 | 9 | 0 | 12 |
| base-b-test-1-npu-a3 / run (0) | 20 | 7 | 0 | 13 |
| base-b-test-16-npu-a3 / run (0) | 21 | 4 | 0 | 17 |
| base-b-test-2-npu-a3 / run (0) | 21 | 7 | 0 | 14 |
| base-b-test-4-npu-a3 / run (0) | 21 | 6 | 0 | 15 |
| base-b-test-4-npu-a3 / run (1) | 21 | 7 | 0 | 14 |
| base-b-test-8-npu-a3 / run (0) | 21 | 7 | 0 | 14 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21 | 7 | 0 | 14 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 21 | 7 | 0 | 14 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 21 | 7 | 0 | 14 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 21 | 8 | 0 | 13 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 7 | 1 | 0 | 6 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 7 | 0 | 0 | 7 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 7 | 2 | 0 | 5 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8 | 0 | 0 | 8 |
| multimodal-gen-test-1-npu-a3 | 22 | 14 | 1 | 7 |

---


## [Run #31497751984](https://github.com/sgl-project/sglang/actions/runs/31497751984)
- **分支**: `codex/cpu-offload-components-clean`
- **总耗时**: 41.9min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31497751984

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31497751984/job/93799736766) |


## [Run #31495329896](https://github.com/sgl-project/sglang/actions/runs/31495329896)
- **分支**: `gdn-ratio3-flashinfer-prefill-sm90`
- **总耗时**: 11.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31495329896

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803676) |
| base-b-test-8-npu-a3 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803857) |
| base-b-test-4-npu-a3 / run (1) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803927) |
| base-b-test-2-npu-a3 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803987) |
| base-a-test-1-npu-a2 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803996) |
| base-b-test-1-npu-a3 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804030) |
| base-b-test-4-npu-a3 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804081) |
| base-b-test-16-npu-a3 / run (0) | 9.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804137) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804420) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804451) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804479) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804483) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储文件缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803676

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803857

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803927

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803987

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791803996

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804030

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804081

- **base-b-test-16-npu-a3 / run (0)**: 作业在下载或访问某个blob时，返回BlobNotFound错误，可能是构建产物或依赖文件未上传或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804137

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804420

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804451

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804479

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31495329896/job/93791804483


## [Run #31494765760](https://github.com/sgl-project/sglang/actions/runs/31494765760)
- **分支**: `main`
- **总耗时**: 11.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31494765760

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696091) |
| base-a-test-1-npu-a2 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696276) |
| base-b-test-4-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696356) |
| base-b-test-2-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696401) |
| base-b-test-16-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696567) |
| base-b-test-4-npu-a3 / run (1) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696644) |
| base-b-test-8-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696650) |
| base-b-test-1-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696983) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789697075) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789697132) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789697260) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789697361) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js 20弃用警告和上传artifact时未找到文件。无法判断具体失败原因，可能为测试未运行或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696091

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或依赖文件在 Azure Blob 存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696276

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696356

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696401

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696567

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696644

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696650

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789696983

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789697075

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789697132

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789697260

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494765760/job/93789697361


## [Run #31494226228](https://github.com/sgl-project/sglang/actions/runs/31494226228)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 9.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31494226228

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047566) |
| base-a-test-1-npu-a2 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047746) |
| base-b-test-2-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047749) |
| base-b-test-4-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047752) |
| base-b-test-16-npu-a3 / run (0) | 8.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047813) |
| base-b-test-4-npu-a3 / run (1) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047815) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047844) |
| base-b-test-1-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047891) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788048114) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788048116) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788048220) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788048447) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047566

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是构建产物或依赖未正确上传，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047746

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047749

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047752

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047813

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的工件/缓存文件已被删除或路径错误，属于外部存储依赖问题，需检查相关 blob 路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047815

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到。这可能是由于 CI 配置中引用的文件被删除、路径错误或上传失败，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047844

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788047891

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788048114

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788048116

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重、数据或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788048220

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31494226228/job/93788048447


## [Run #31493433076](https://github.com/sgl-project/sglang/actions/runs/31493433076)
- **分支**: `gdn-ratio3-flashinfer-prefill-sm90`
- **总耗时**: 23.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31493433076

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 12.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785320830) |
| base-b-test-2-npu-a3 / run (0) | 22.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321453) |
| base-b-test-8-npu-a3 / run (0) | 22.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321484) |
| base-b-test-16-npu-a3 / run (0) | 22.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321561) |
| base-a-test-1-npu-a2 / run (0) | 22.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321571) |
| base-b-test-1-npu-a3 / run (0) | 22.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321683) |
| base-b-test-4-npu-a3 / run (0) | 22.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321737) |
| base-b-test-4-npu-a3 / run (1) | 22.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321771) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 22.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785322661) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785322715) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 22.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785322761) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 22.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785322854) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本弃用警告及上传失败产物（无文件），未包含multimodal测试执行过程或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785320830

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321453

- **base-b-test-8-npu-a3 / run (0)**: 作业在下载或访问Azure Blob存储中的某个blob时失败，返回BlobNotFound错误。这通常是因为文件被删除、路径错误或存储配置变更，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321484

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理或路径配置错误，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321561

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存、模型文件或日志缺失，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321571

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源（如模型权重、数据集或缓存文件）已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321683

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321737

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785321771

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785322661

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785322715

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785322761

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493433076/job/93785322854


## [Run #31493026763](https://github.com/sgl-project/sglang/actions/runs/31493026763)
- **分支**: `pd-fix-2`
- **总耗时**: 21.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31493026763

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974067) |
| base-b-test-4-npu-a3 / run (1) | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974072) |
| base-a-test-1-npu-a2 / run (0) | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974081) |
| base-b-test-8-npu-a3 / run (0) | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974127) |
| base-b-test-4-npu-a3 / run (0) | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974140) |
| base-b-test-2-npu-a3 / run (0) | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974190) |
| base-b-test-1-npu-a3 / run (0) | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974205) |
| base-b-test-16-npu-a3 / run (0) | 20.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974210) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974415) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974464) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974475) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974563) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974067

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974072

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974081

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974127

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974140

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974190

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974205

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974210

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974415

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被删除或链接失效，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974464

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974475

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31493026763/job/93783974563


## [Run #31492513829](https://github.com/sgl-project/sglang/actions/runs/31492513829)
- **分支**: `dots-note-for-sglang`
- **总耗时**: 9.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31492513829

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307753) |
| base-b-test-16-npu-a3 / run (0) | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307853) |
| base-b-test-2-npu-a3 / run (0) | 8.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307896) |
| base-b-test-8-npu-a3 / run (0) | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307901) |
| base-a-test-1-npu-a2 / run (0) | 8.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307940) |
| base-b-test-4-npu-a3 / run (1) | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307955) |
| base-b-test-4-npu-a3 / run (0) | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308082) |
| base-b-test-1-npu-a3 / run (0) | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308316) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308332) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308392) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308444) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308473) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现测试执行或失败断言，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307753

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307853

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象缺失，可能是构建产物未上传、路径错误或存储被清理，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307896

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307901

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是缓存、依赖或日志文件缺失，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307940

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799307955

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308082

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308316

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308332

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308392

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308444

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492513829/job/93799308473


## [Run #31492376732](https://github.com/sgl-project/sglang/actions/runs/31492376732)
- **分支**: `main`
- **总耗时**: 20.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31492376732

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 19.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834033) |
| base-b-test-8-npu-a3 / run (0) | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834036) |
| base-b-test-4-npu-a3 / run (1) | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834047) |
| base-b-test-16-npu-a3 / run (0) | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834075) |
| base-b-test-1-npu-a3 / run (0) | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834092) |
| base-b-test-4-npu-a3 / run (0) | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834114) |
| base-a-test-1-npu-a2 / run (0) | 19.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834133) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834401) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834454) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834460) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834500) |

- **base-b-test-2-npu-a3 / run (0)**: 作业运行期间尝试下载或访问的 blob 资源未找到（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834033

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境配置或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834036

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834047

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834075

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834092

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834114

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834133

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834401

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834454

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834460

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31492376732/job/93781834500


## [Run #31492030090](https://github.com/sgl-project/sglang/actions/runs/31492030090)
- **分支**: `codex/diffusion-auto-layerwise-policy`
- **总耗时**: 38.5min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31492030090

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31492030090/job/93780663543) |


## [Run #31491657646](https://github.com/sgl-project/sglang/actions/runs/31491657646)
- **分支**: `hisparse_mtp_kernel`
- **总耗时**: 48.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31491657646

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779485863) |
| base-b-test-4-npu-a3 / run (0) | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779485908) |
| base-b-test-4-npu-a3 / run (1) | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779485912) |
| base-b-test-16-npu-a3 / run (0) | 47.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486003) |
| base-b-test-8-npu-a3 / run (0) | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486017) |
| base-b-test-1-npu-a3 / run (0) | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486021) |
| base-b-test-2-npu-a3 / run (0) | 47.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486032) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486474) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486493) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486564) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 47.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486572) |

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779485863

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779485908

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779485912

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储账户中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486003

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是上游作业未成功上传或存储配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486017

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查上传步骤或资源生命周期。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486021

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486032

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486474

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486493

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486564

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491657646/job/93779486572


## [Run #31491110682](https://github.com/sgl-project/sglang/actions/runs/31491110682)
- **分支**: `diffusion-ideogram-rope-silu-fusion`
- **总耗时**: 51.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31491110682

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | 环境问题 | 容器内安装Rust时下载rustup-init超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815763) |
| base-b-test-16-npu-a3 / run (0) | 50.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815814) |
| base-b-test-4-npu-a3 / run (0) | 50.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815889) |
| base-b-test-1-npu-a3 / run (0) | 50.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815897) |
| base-b-test-4-npu-a3 / run (1) | 50.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815991) |
| base-b-test-8-npu-a3 / run (0) | 50.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816006) |
| base-b-test-2-npu-a3 / run (0) | 50.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816093) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 50.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816441) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 50.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816463) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 50.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816485) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 50.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816501) |

- **base-a-test-1-npu-a2 / run (0)**: 作业在自定义容器中执行，因未预装Rust，尝试从内部缓存服务下载rustup-init，但下载过程超时（约2分钟无响应），导致容器执行失败，属于环境依赖获取问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815763

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或缓存缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815814

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815889

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815897

- **base-b-test-4-npu-a3 / run (1)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是上传失败、路径错误或资源被清理，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815991

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816006

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816093

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816441

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816463

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816485

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777816501

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31491110682/job/93777815602) |


## [Run #31490838473](https://github.com/sgl-project/sglang/actions/runs/31490838473)
- **分支**: `codex/cpu-offload-components-clean`
- **总耗时**: 34.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31490838473

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 26.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31490838473/job/93776793588) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js版本弃用警告和上传artifact时未找到文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31490838473/job/93776793588


## [Run #31489956661](https://github.com/sgl-project/sglang/actions/runs/31489956661)
- **分支**: `main`
- **总耗时**: 30.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31489956661

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 29.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929193) |
| base-a-test-1-npu-a2 / run (0) | 29.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929195) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929214) |
| base-b-test-4-npu-a3 / run (1) | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929217) |
| base-b-test-1-npu-a3 / run (0) | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929294) |
| base-b-test-2-npu-a3 / run (0) | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929303) |
| base-b-test-8-npu-a3 / run (0) | 29.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929402) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929650) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929766) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929862) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929894) |

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问的 blob 资源未找到（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929193

- **base-a-test-1-npu-a2 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/日志文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929195

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象缺失，可能是构建产物未上传、路径错误或存储被清理，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929214

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929217

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929294

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929303

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929402

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929650

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929766

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929862

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773929894

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31489956661/job/93773928960) |


## [Run #31487148563](https://github.com/sgl-project/sglang/actions/runs/31487148563)
- **分支**: `fix-pack-topk-int64`
- **总耗时**: 177.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31487148563

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 72.1min | 代码错误 | NPU PD分离测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118053) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93789132344) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.5min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93794693465) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93800221609) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93813541858) |

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（exit code 1），其余3个测试通过。该测试耗时仅325秒，非超时，属于功能测试失败，可能是代码逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118053

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1126秒后退出码1，0/1通过，属于性能指标未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93789132344

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，被判定为根因失败，导致本作业（16-npu-a3）在启动前被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93794693465

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93800221609

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3，本作业因快速失败机制被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93813541858

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765117948) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765117957) |
| base-a-test-1-npu-a2 / run (0) | 10.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765117982) |
| base-b-test-4-npu-a3 / run (0) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118008) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118011) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118067) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118198) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 86.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118217) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118248) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31487148563/job/93765118275) |


## [Run #31486917906](https://github.com/sgl-project/sglang/actions/runs/31486917906)
- **分支**: `mmangkad/flashinfer-0.6.17rc1-kimi-k3`
- **总耗时**: 182.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31486917906

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 66.7min | 代码错误 | NPU PD分离测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512237) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.6min | 超时 | 性能测试用例执行超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93778917607) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 2.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93783595224) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-NPU perf）失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93789719687) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93813889235) |

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（exit code 1），其余3个测试通过。该测试在361秒内失败，可能是代码逻辑或环境配置问题导致，需查看具体测试日志定位原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512237

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1073秒后退出码1，超过预估3600秒限制，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93778917607

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（16-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93783595224

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 本作业在启动前的PR健康检查中检测到根因失败作业base-c-test-perf-8-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93789719687

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业（base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3），本作业被Fast-fail机制跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93813889235

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512100) |
| base-b-test-1-npu-a3 / run (0) | 24.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512224) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512260) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512266) |
| base-b-test-2-npu-a3 / run (0) | 20.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512276) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512368) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 123.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512501) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512536) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512585) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512610) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486917906/job/93764512779) |


## [Run #31486012963](https://github.com/sgl-project/sglang/actions/runs/31486012963)
- **分支**: `diffusion-ideogram-rope-silu-fusion`
- **总耗时**: 67.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31486012963

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 5.1min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603113) |
| base-b-test-4-npu-a3 / run (0) | 3.3min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603132) |
| base-b-test-2-npu-a3 / run (0) | 8.2min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603138) |
| base-b-test-16-npu-a3 / run (0) | 66.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603168) |
| base-b-test-8-npu-a3 / run (0) | 3.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603178) |
| base-b-test-1-npu-a3 / run (0) | 8.3min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603253) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.3min | 环境问题 | 自定义容器执行失败，NPU环境或容器配置异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603621) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.2min | 环境问题 | 自定义容器执行失败，无法启动测试环境 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603662) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603677) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 9.4min | 超时 | 性能测试因Scheduler watchdog超时失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93774692119) |

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在加载模型权重时，runner报错“Executing the custom container implementation failed”，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603113

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动NPU测试容器时失败，错误信息为"Executing the custom container implementation failed"，属于自托管runner环境问题，与测试代码无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603132

- **base-b-test-2-npu-a3 / run (0)**: 日志显示在加载模型权重过程中，自定义容器实现执行失败（Executing the custom container implementation failed），提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603138

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603168

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动测试前执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603178

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在模型捕获批次过程中出现torch_npu相关警告，随后报错"Executing the custom container implementation failed"，提示联系self hosted runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603253

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示服务启动后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU容器环境或基础设施问题，非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603621

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段因自定义容器实现执行失败而终止，错误信息提示联系自托管runner管理员，属于基础设施或容器配置问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603662

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603677

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示Scheduler watchdog timeout (300s)，TP4 EP4调度器在等待队列时超时，导致容器执行失败，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93774692119

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603069) |
| base-a-test-1-npu-a2 / run (0) | 12.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603182) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31486012963/job/93761603616) |


## [Run #31485916694](https://github.com/sgl-project/sglang/actions/runs/31485916694)
- **分支**: `lm-head-opt`
- **总耗时**: 144.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31485916694

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 64.8min | 代码错误 | NPU PD分离测试用例执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248390) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | 性能回归 | NPU性能测试未通过，minimax_m2_5 w8a8测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93774182043) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因同PR中另一个作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93779833042) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93784493749) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 环境问题 | 健康检查检测到其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93796007288) |

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试文件运行退出码为1，耗时356秒，导致整个作业失败。其他三个测试均通过，问题集中在PD分离功能测试上。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248390

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1119秒后失败，属于性能测试用例，可能未达到预期性能指标或出现性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93774182043

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（16-npu）因快速失败机制被跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93779833042

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，本作业作为级联失败被过滤，最终因根因作业失败而快速失败，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93784493749

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为base-b-test-16-npu-a3和base-c-test-perf-8-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93796007288

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248309) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248401) |
| base-a-test-1-npu-a2 / run (0) | 8.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248422) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248436) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248443) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248445) |
| base-b-test-4-npu-a3 / run (0) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248531) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248724) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248755) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248793) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485916694/job/93761248805) |


## [Run #31485463912](https://github.com/sgl-project/sglang/actions/runs/31485463912)
- **分支**: `main_8.5`
- **总耗时**: 59.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31485463912

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803083) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803109) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803141) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中的lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803163) |
| base-a-test-1-npu-a2 / run (0) | 1.2min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803173) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803176) |
| base-b-test-4-npu-a3 / run (0) | 0.6min | 其他 | 健康检查失败：lint检查未通过导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803187) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.9min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803859) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 环境问题 | PR健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803886) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | 健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803922) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.1min | 其他 | 健康检查失败：lint检查未通过，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803931) |

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到lint检查状态为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803083

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803109

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803141

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业在运行测试前即终止，退出码为1。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803163

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803173

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803176

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到PR的lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际NPU测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803187

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查结论为failure，触发fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803859

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查结论为failure，触发了fast-fail机制，作业在运行实际测试前即被终止，属于前置检查失败而非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803886

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查状态为failure，触发fast-fail机制，作业未进入实际测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803922

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业立即失败，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759803931

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 39.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31485463912/job/93759802896) |


## [Run #31484207064](https://github.com/sgl-project/sglang/actions/runs/31484207064)
- **分支**: `agent/optimize-sm120-ernie-norm-fusion`
- **总耗时**: 39.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31484207064

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (1) | 3.5min | 环境问题 | 自定义容器执行失败，NPU测试环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916831) |
| base-b-test-4-npu-a3 / run (0) | 3.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916872) |
| base-b-test-8-npu-a3 / run (0) | 3.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916887) |
| base-b-test-2-npu-a3 / run (0) | 2.3min | 环境问题 | 下载依赖包时容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916907) |
| base-b-test-16-npu-a3 / run (0) | 3.5min | 环境问题 | 自定义容器执行失败，NPU测试未开始即中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916915) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.5min | 环境问题 | 自定义容器执行失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917352) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 38.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917389) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 38.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917442) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 38.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917444) |

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动测试时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916831

- **base-b-test-4-npu-a3 / run (0)**: 作业在运行自定义容器实现时失败，错误为'Executing the custom container implementation failed'，属于runner环境或容器配置问题，非测试代码本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916872

- **base-b-test-8-npu-a3 / run (0)**: 作业在运行测试前，执行自定义容器实现时失败，错误提示联系runner管理员，属于基础设施环境问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916887

- **base-b-test-2-npu-a3 / run (0)**: 作业在下载ops-transformer依赖包时，自定义容器实现执行失败，提示联系自托管runner管理员，可能是网络或容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916907

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动测试前，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，与测试代码无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916915

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装Rust工具链后，执行自定义容器实现时出错，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917352

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917389

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917442

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755917444

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 8.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31484207064/job/93755916930) |


## [Run #31484120985](https://github.com/sgl-project/sglang/actions/runs/31484120985)
- **分支**: `rope_mova_unification`
- **总耗时**: 34.4min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31484120985

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31484120985/job/93755581261) |


## [Run #31483724651](https://github.com/sgl-project/sglang/actions/runs/31483724651)
- **分支**: `agent/optimize-sm120-ernie-norm-fusion`
- **总耗时**: 7.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31483724651

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 1.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350683) |
| base-b-test-2-npu-a3 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350859) |
| base-b-test-8-npu-a3 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350878) |
| base-b-test-16-npu-a3 / run (0) | 6.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350921) |
| base-b-test-1-npu-a3 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350934) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350964) |
| base-b-test-4-npu-a3 / run (1) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351107) |
| base-b-test-4-npu-a3 / run (0) | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351174) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351277) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351308) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351325) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351351) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350683

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350859

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350878

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350921

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350934

- **base-a-test-1-npu-a2 / run (0)**: 作业在下载或访问某个blob时，返回BlobNotFound错误，可能是CI配置中引用的文件被删除或路径错误，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754350964

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351107

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351174

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351277

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351308

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351325

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31483724651/job/93754351351


## [Run #31483499928](https://github.com/sgl-project/sglang/actions/runs/31483499928)
- **分支**: `codex/cpu-offload-components-clean`
- **总耗时**: 28.7min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31483499928

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31483499928/job/93753600058) |


## [Run #31915187512](https://github.com/sgl-project/sglang/actions/runs/31915187512)
- **分支**: `main`
- **总耗时**: 141.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31915187512

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 8.0min | 代码错误 | HiCache MLA测试用例执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903848) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.9min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95086477124) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 38.0min | 性能回归 | NPU性能测试中qwen3_235b_w8a8用例失败，仅1/4通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95088257363) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 18.3min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95098386053) |

- **base-b-test-4-npu-a3 / run (0)**: test_npu_hicache_mla.py测试文件运行失败，退出码为1，导致整个作业失败。该测试属于NPU基础功能测试，可能涉及HiCache MLA相关功能实现问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903848

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1049秒后退出码1，0/1通过，属于性能测试未达标或执行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95086477124

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例退出码1，耗时1439秒，可能因性能未达阈值或运行错误导致，其余用例通过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95088257363

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示测试运行中容器突然终止，报错'Executing the custom container implementation failed'，属于runner或容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95098386053

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 21.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903641) |
| base-b-test-8-npu-a3 / run (0) | 11.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903716) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903770) |
| base-b-test-2-npu-a3 / run (0) | 27.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903780) |
| base-b-test-4-npu-a3 / run (1) | 25.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903792) |
| base-b-test-16-npu-a3 / run (0) | 69.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903798) |
| base-b-test-1-npu-a3 / run (0) | 47.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903897) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903945) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 120.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085903964) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085904020) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95085904028) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 17.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915187512/job/95090260643) |


## [Run #31915127701](https://github.com/sgl-project/sglang/actions/runs/31915127701)
- **分支**: `lsyin/shared-read-default-pre-replay`
- **总耗时**: 130.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31915127701

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.5min | 性能回归 | NPU性能测试未达预期，minimax_m2_5测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95086157849) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 41.8min | 性能回归 | 性能测试用例qwen3_235b_w8a8_8p_in3k5_out1k5_50ms失败，退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95088515681) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 46.6min | 精度回归 | NPU性能测试中qwen3_vl_8b_thinking_1p_mmmu测试失败，0/6用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95094372275) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1142秒后退出码为1，0/1测试通过，属于性能指标未达标导致的失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95086157849

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该用例为性能测试，执行耗时1408秒，远超其他用例（492秒、386秒），且未通过，疑似性能未达标或运行异常，导致整体作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95088515681

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 测试test_npu_qwen3_vl_8b_thinking_1p_mmmu.py返回退出码1，耗时2558秒，所有6个测试均未通过，属于精度回归问题，需检查模型输出或数据精度。
  链接: https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95094372275

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783005) |
| base-b-test-2-npu-a3 / run (0) | 19.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783008) |
| multimodal-gen-test-1-npu-a3 | 22.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783016) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783017) |
| base-b-test-4-npu-a3 / run (1) | 15.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783020) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783022) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783024) |
| base-b-test-16-npu-a3 / run (0) | 46.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783040) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783150) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 82.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783169) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783174) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95085783177) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31915127701/job/95089596314) |


## [Run #31914132334](https://github.com/sgl-project/sglang/actions/runs/31914132334)
- **分支**: `feat/triton-sparse-mla`
- **总耗时**: 91.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31914132334

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.1min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95084102338) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 60.0min | 性能回归 | NPU性能测试中qwen3_235b_a22b用例失败，其余用例通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95086053877) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95087997078) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95092570173) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1133秒后退出码为1，属于性能测试未通过，可能因模型推理性能未达到预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95084102338

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中qwen3_235b_a22b的w8a8_8p_in3k5_out1k5_50ms用例执行失败（exit code 1），耗时1455秒，其他三个用例均通过，疑似该模型性能未达预期或运行异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95086053877

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（4-npu）被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95087997078

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业（base-c-test-perf-8/16-npu-a3），本作业被快速失败机制跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95092570173

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474232) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474277) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474289) |
| base-b-test-2-npu-a3 / run (0) | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474298) |
| base-b-test-4-npu-a3 / run (0) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474322) |
| base-b-test-4-npu-a3 / run (1) | 15.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474358) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474374) |
| base-b-test-16-npu-a3 / run (0) | 55.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474387) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474476) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474489) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474494) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31914132334/job/95083474548) |


## [Run #31910955003](https://github.com/sgl-project/sglang/actions/runs/31910955003)
- **分支**: `feat/roofline_annotations`
- **总耗时**: 301.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31910955003

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.0min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95077552457) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，因同PR中另一个作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95080357992) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（perf-8-npu-a3）已失败，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95089349413) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1138秒后失败，该测试为性能测试，预计时间3600秒，实际未通过，可能因性能指标未达标或执行错误导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95077552457

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业本身未执行，因健康检查发现同PR的base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，导致本作业被跳过并报错。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95080357992

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业为base-c-test-perf-8-npu-a3，本作业（perf-2）被级联跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95089349413

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075827937) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828054) |
| base-b-test-8-npu-a3 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828058) |
| base-b-test-4-npu-a3 / run (0) | 31.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828121) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828141) |
| base-b-test-16-npu-a3 / run (0) | 53.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828149) |
| base-b-test-2-npu-a3 / run (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828156) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828217) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 128.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828294) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828310) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828358) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95075828363) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 265.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31910955003/job/95079613749) |


---
*Auto-generated by npu_pr_monitor.py*