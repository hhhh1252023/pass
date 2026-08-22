# NPU CI 执行监控
**生成时间**: 2026-08-22 10:35 UTC
**分析 Run 数**: 44

---

## 📊 本次执行总结

- **成功 Job 数**: 205
- **失败 Run 数**: 44
- **成功 Job 平均耗时**: 24.9min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 130.2min | #32554954256 | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596344) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 125.5min | #32541519730 | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417610) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 113.0min | #32542372361 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755806) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.7min | #32552577485 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680348) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 103.5min | #32552872558 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982420043) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.4min | #32548508586 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971250151) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.4min | #32545207473 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383361) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.3min | #32552911090 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472919) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.4min | #32549339861 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409790) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.6min | #32543410873 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640444) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 44 | 0 | 27 | 0 | 17 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 12 | 0 | 0 | 11 | 1 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 13 | 5 | 0 | 8 | 0 |
| base-b-test-16-npu-a3 / run (0) | 32 | 12 | 0 | 5 | 15 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 32 | 12 | 0 | 5 | 15 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17 | 12 | 0 | 5 | 0 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 32 | 13 | 0 | 4 | 15 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 14 | 10 | 0 | 4 | 0 |
| base-b-test-4-npu-a3 / run (0) | 32 | 15 | 0 | 3 | 14 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 32 | 14 | 0 | 3 | 15 |
| base-b-test-1-npu-a3 / run (0) | 32 | 16 | 0 | 2 | 14 |
| base-b-test-4-npu-a3 / run (1) | 32 | 16 | 0 | 2 | 14 |
| base-b-test-2-npu-a3 / run (0) | 32 | 16 | 0 | 2 | 14 |
| base-a-test-1-npu-a2 / run (0) | 32 | 31 | 0 | 1 | 0 |
| base-b-test-8-npu-a3 / run (0) | 32 | 16 | 0 | 1 | 15 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32541519730 | `cheng/gc-cut-1` | 354.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32541519730) |
| #32543410873<br>[#32577 [AMD] DeepSeek-V4: add aiter fused mHC post+pre with cross-layer boundary dispatch](https://github.com/sgl-project/sglang/pull/32577) | `amd/dsv4-aiter-fused-mhc-cross-layer` | 287.7min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32543410873) |
| #32545207473<br>[#33863 [Feature] PP Support PD + DSpark](https://github.com/sgl-project/sglang/pull/33863) | `deepseek_v4_dspark_suppport_pp_pd` | 279.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32545207473) |
| #32545378215<br>[#35760 [Perf] Tune the W4AFP8 DeepEP low-latency requant launch geometry](https://github.com/sgl-project/sglang/pull/35760) | `perf/w4a8-ll-requant-tuning` | 275.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32545378215) |
| #32542372361<br>[#35758 qwen 3.8 rebase](https://github.com/sgl-project/sglang/pull/35758) | `qwen-qiaolin` | 262.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32542372361) |
| #32543423862<br>[#33576  [AMD] Add Work-Centric (Lean) Attention: a persistent-CTA decode kernel for long-context serving](https://github.com/sgl-project/sglang/pull/33576) | `wca-rebased` | 207.1min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32543423862) |
| #32552872558<br>[#33576  [AMD] Add Work-Centric (Lean) Attention: a persistent-CTA decode kernel for long-context serving](https://github.com/sgl-project/sglang/pull/33576) | `wca-rebased` | 200.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32552872558) |
| #32551034018<br>[#33778 Avoid materializing GDN QKV tensors during target verification](https://github.com/sgl-project/sglang/pull/33778) | `perf/gdn-strided-target-verify` | 197.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32551034018) |
| #32554954256<br>[#34565 [Unified Tree] Support Branching-Point Caching for the SWA Component](https://github.com/sgl-project/sglang/pull/34565) | `cjc/unified-swa-branching` | 195.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32554954256) |
| #32552911090<br>[#35914 Fix nullable GLM tool argument parsing](https://github.com/sgl-project/sglang/pull/35914) | `fix/glm-nullable-tool-arguments` | 190.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32552911090) |
| #32552577485<br>[#33279 [FEAT] Weight Daemon abstraction](https://github.com/sgl-project/sglang/pull/33279) | `lsy` | 189.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32552577485) |
| #32549339861<br>[#33113 [AMD] Add AITER HIP backend for packed GDN decode on gfx950](https://github.com/sgl-project/sglang/pull/33113) | `feat/aiter-gfx950-gdn-decode-backend` | 188.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32549339861) |
| #32548508586<br>[#33829 [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding](https://github.com/sgl-project/sglang/pull/33829) | `dots-note-for-sglang` | 183.4min | 2 | multimodal-gen-test-1-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32548508586) |
| #32550200986<br>[#32754 [AMD] Enable gfx1250 Support](https://github.com/sgl-project/sglang/pull/32754) | `amd_helios` | 180.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32550200986) |
| #32548755133<br>[#35933 [HiCache] Clamp tombstoned SWA locs in UnifiedSWAKVPool translation](https://github.com/sgl-project/sglang/pull/35933) | `zqx/hicache-swa-translate-sentinel-clamp` | 167.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32548755133) |
| #32548236937 | `feat/mapped-layer-courier` | 152.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32548236937) |
| #32548700954 | `feature/load-reporter` | 135.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32548700954) |
| #32542984477<br>[#35769 Fix buffer-mode HiCache load-back ownership races; add optional prefetch anchor lock](https://github.com/sgl-project/sglang/pull/35769) | `hicache-buffer-anchor-lock-oss` | 123.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32542984477) |
| #32543681823<br>[#33279 [FEAT] Weight Daemon abstraction](https://github.com/sgl-project/sglang/pull/33279) | `lsy` | 98.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32543681823) |
| #32548975515 | `dcp-trtllm-mla-decode` | 95.7min | 7 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32548975515) |
| #32544030553<br>[#35939 [diffusion] feat: resolve hub component subfolders](https://github.com/sgl-project/sglang/pull/35939) | `codex/diffusion-component-hub-subfolders` | 80.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32544030553) |
| #32544255126 | `feat/mapped-layer-courier` | 74.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32544255126) |
| #32544377288 | `feat/layerwise-all-literal` | 73.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32544377288) |
| #32551251825 | `fix/vae-gate-local-root` | 72.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32551251825) |
| #32543451029<br>[#35922 [diffusion] feat: add maybe_record_function profiler spans for request phases](https://github.com/sgl-project/sglang/pull/35922) | `feat/diffusion-maybe-record-function` | 72.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32543451029) |
| #32549394976<br>[#33279 [FEAT] Weight Daemon abstraction](https://github.com/sgl-project/sglang/pull/33279) | `lsy` | 71.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32549394976) |
| #32545220119<br>[#35749 [diffusion] read the next mapped layer ahead instead of faulting on it](https://github.com/sgl-project/sglang/pull/35749) | `feat/mapped-weight-readahead` | 68.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32545220119) |
| #32545065555<br>[#35832 [diffusion] fix a refit KeyError on mapped weights, and stop claiming strides the reload discards](https://github.com/sgl-project/sglang/pull/35832) | `fix/mapped-weight-metadata` | 67.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32545065555) |
| #32550981238<br>[#35674 [diffusion] docs+skill: which components to stream under layerwise offload](https://github.com/sgl-project/sglang/pull/35674) | `codex/diffusion-t5-bnb-text-encoder` | 58.8min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32550981238) |
| #32546336331<br>[#35747 Add sampling observer auxiliary output hooks](https://github.com/sgl-project/sglang/pull/35747) | `main` | 52.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32546336331) |
| #32553573136<br>[#35674 [diffusion] docs+skill: which components to stream under layerwise offload](https://github.com/sgl-project/sglang/pull/35674) | `codex/diffusion-t5-bnb-text-encoder` | 44.4min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32553573136) |
| #32550595701<br>[#35939 [diffusion] feat: resolve hub component subfolders](https://github.com/sgl-project/sglang/pull/35939) | `main` | 32.9min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32550595701) |
| #32547515476 | `dcp-trtllm-mla-decode` | 32.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32547515476) |
| #32548707270<br>[#35769 Fix buffer-mode HiCache load-back ownership races; add optional prefetch anchor lock](https://github.com/sgl-project/sglang/pull/35769) | `main` | 31.0min | 11 | base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32548707270) |
| #32544641005<br>[#32984 [MLX] Upgrade to Torch 2.13/MLX 0.32+ and redesign the Torch-MLX tensor bridge](https://github.com/sgl-project/sglang/pull/32984) | `main` | 27.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32544641005) |
| #32543329727<br>[#32984 [MLX] Upgrade to Torch 2.13/MLX 0.32+ and redesign the Torch-MLX tensor bridge](https://github.com/sgl-project/sglang/pull/32984) | `mlx-032-torch-213-bridge` | 27.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32543329727) |
| #32542789457<br>[#35921 [Fix] Read the granite sinks dtype from the exec bag, not the legacy global shim](https://github.com/sgl-project/sglang/pull/35921) | `main` | 18.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32542789457) |
| #32548618621<br>[#33279 [FEAT] Weight Daemon abstraction](https://github.com/sgl-project/sglang/pull/33279) | `lsy` | 17.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32548618621) |
| #32544682431 | `yuychang/k3-kda-inproj-fusion` | 13.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32544682431) |
| #32550113139<br>[#35890 fix(disagg): PD transfer-failure injection was silently inert](https://github.com/sgl-project/sglang/pull/35890) | `main` | 10.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32550113139) |
| #32545921948<br>[#35888 Support CPU offload for mxfp8 KV cache](https://github.com/sgl-project/sglang/pull/35888) | `main` | 9.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32545921948) |
| #32544290089<br>[#35918 [DeepSeek V4] Add W4A4 MegaMoE server flag](https://github.com/sgl-project/sglang/pull/35918) | `main` | 7.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32544290089) |
| #32543977890<br>[#35867 [diffusion] refactor: hand out pinned host memory per layer](https://github.com/sgl-project/sglang/pull/35867) | `main` | 6.9min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32543977890) |
| #32553062710<br>[#35729 [diffusion] Enable SANA-Video breakable CUDA graphs](https://github.com/sgl-project/sglang/pull/35729) | `main` | 6.3min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32553062710) |

---


## [Run #32554954256](https://github.com/sgl-project/sglang/actions/runs/32554954256)
- **分支**: `cjc/unified-swa-branching`
- **总耗时**: 195.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32554954256

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.7min | 环境问题 | Runner 收到关闭信号导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596118) |
| base-b-test-4-npu-a3 / run (0) | 11.2min | 环境问题 | 测试进程被系统以退出码137（OOM）终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596190) |
| base-b-test-16-npu-a3 / run (0) | 13.0min | 环境问题 | NPU容器在测试运行中被强制终止，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596215) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.9min | 环境问题 | 作业因容器被杀死（exit code 137）而失败，可能是内存不足或OOM。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596272) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.1min | 环境问题 | 容器内安装依赖时进程被OOM杀死，退出码137。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596309) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 9.4min | 环境问题 | NPU性能测试进程被系统OOM杀死，退出码137 | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96994912239) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/97008655307) |

- **multimodal-gen-test-1-npu-a3**: 作业在加载 FLUX.1-dev 模型时，runner 收到 shutdown 信号（exit code 130），可能是自托管 runner 被手动取消或服务停止，并非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596118

- **base-b-test-4-npu-a3 / run (0)**: 日志显示测试运行中命令被非零退出码137终止，137通常表示进程被SIGKILL（内存溢出），且发生在NPU测试执行期间，属于环境资源不足问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596190

- **base-b-test-16-npu-a3 / run (0)**: 日志显示测试进程在加载模型时被系统以退出码137（SIGKILL）终止，随后容器丢失，可能是由于内存不足或资源限制导致OOM被杀。
  链接: https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596215

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示运行过程中容器突然终止，退出码137通常表示OOM或被系统杀死。后续出现“container not found”错误，表明容器已不存在。这属于环境资源问题，而非代码或精度问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596272

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 在安装evalscope等依赖时，pip安装过程内存不足，导致进程被系统OOM killer终止（exit code 137），属于环境资源限制问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596309

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示在CUDA graph捕获阶段，进程因内存不足被内核OOM killer终止（exit code 137），随后容器也异常退出。这属于NPU环境资源限制问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96994912239

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到多个根因作业失败（如base-b-test-4-npu-a3等），触发fast-fail逻辑，本作业未实际运行即被取消，属于依赖的上游失败导致的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32554954256/job/97008655307

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 21.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596131) |
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596151) |
| base-b-test-8-npu-a3 / run (0) | 7.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596160) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596182) |
| base-b-test-2-npu-a3 / run (0) | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596210) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596322) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 130.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32554954256/job/96987596344) |


## [Run #32553573136](https://github.com/sgl-project/sglang/actions/runs/32553573136)
- **分支**: `codex/diffusion-t5-bnb-text-encoder`
- **总耗时**: 44.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32553573136

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 17.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32553573136/job/96984203250) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤（未找到失败文件），未展示multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32553573136/job/96984203250


## [Run #32553062710](https://github.com/sgl-project/sglang/actions/runs/32553062710)
- **分支**: `main`
- **总耗时**: 6.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32553062710

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32553062710/job/96982867310) |

- **multimodal-gen-test-1-npu-a3**: 作业在下载依赖或数据时，请求的Azure Blob返回BlobNotFound错误，可能是文件被删除、路径错误或存储配置变更，属于环境或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32553062710/job/96982867310


## [Run #32552911090](https://github.com/sgl-project/sglang/actions/runs/32552911090)
- **分支**: `fix/glm-nullable-tool-arguments`
- **总耗时**: 190.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32552911090

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472742) |
| base-b-test-16-npu-a3 / run (0) | 6.7min | 环境问题 | Runner 在模型权重加载过程中收到关闭信号，作业被强制终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472871) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472884) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96994073141) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96998601414) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业根因失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/97003084727) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最终仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业仍标记为失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472742

- **base-b-test-16-npu-a3 / run (0)**: 日志显示模型加载到31%时，runner收到shutdown信号，可能是自托管runner被取消或服务停止，导致容器执行失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472871

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到 base-b-test-16-npu-a3 / run (0) 作业失败，被判定为根因，因此本作业被快速失败机制跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472884

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 作业在启动前的PR健康检查中，检测到根因作业base-b-test-16-npu-a3失败，按策略快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96994073141

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因失败作业base-b-test-16-npu-a3，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96998601414

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-b-test-16-npu-a3作业为根因失败，本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552911090/job/97003084727

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472758) |
| base-b-test-1-npu-a3 / run (0) | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472759) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472795) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472823) |
| base-b-test-2-npu-a3 / run (0) | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472830) |
| base-b-test-4-npu-a3 / run (1) | 14.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472850) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472898) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472908) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552911090/job/96983472919) |


## [Run #32552872558](https://github.com/sgl-project/sglang/actions/runs/32552872558)
- **分支**: `wca-rebased`
- **总耗时**: 200.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32552872558

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419826) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 21.9min | 环境问题 | Runner收到关闭信号导致作业中断，非测试本身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982420059) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 20.1min | 其他 | Runner 收到关闭信号，作业被外部终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982420111) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/97003230479) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419826

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示进程以退出码130结束，runner收到shutdown信号，可能是自托管runner被手动停止或服务终止，属于基础设施环境问题，与代码或测试结果无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982420059

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示进程以退出码130结束，runner收到shutdown信号，可能是手动取消或服务停止，并非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982420111

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到 base-c-test-acc-16-npu-a3 和 base-c-test-acc-4-npu-a3 两个根因作业失败，因此本作业被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552872558/job/97003230479

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419869) |
| base-b-test-1-npu-a3 / run (0) | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419883) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419891) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419892) |
| base-b-test-16-npu-a3 / run (0) | 50.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419930) |
| base-b-test-2-npu-a3 / run (0) | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419954) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982419973) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 103.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982420043) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96982420046) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552872558/job/96991881534) |


## [Run #32552577485](https://github.com/sgl-project/sglang/actions/runs/32552577485)
- **分支**: `lsy`
- **总耗时**: 189.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32552577485

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680160) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 20.9min | 环境问题 | Runner 收到关闭信号导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96991671859) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.0min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96993438206) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业因其他根因作业失败被快速跳过，非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96995457996) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/97002025230) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680160

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示测试运行正常，但 runner 在 06:42:41 收到 shutdown 信号，可能是自托管 runner 被停止或取消，导致容器执行失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96991671859

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示该作业在启动前因同一PR中另一个作业（base-c-test-perf-8-npu-a3）失败而被fast-fail跳过，属于依赖的上游作业失败导致的连锁取消，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96993438206

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查发现根因作业base-c-test-perf-8-npu-a3失败，本作业作为级联失败被fast-fail跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96995457996

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，本作业作为级联失败被跳过，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32552577485/job/97002025230

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680165) |
| base-b-test-1-npu-a3 / run (0) | 24.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680174) |
| base-b-test-4-npu-a3 / run (0) | 30.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680185) |
| base-b-test-16-npu-a3 / run (0) | 52.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680204) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680208) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680254) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680271) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680341) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680348) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680354) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32552577485/job/96981680378) |


## [Run #32551251825](https://github.com/sgl-project/sglang/actions/runs/32551251825)
- **分支**: `fix/vae-gate-local-root`
- **总耗时**: 72.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32551251825

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 64.2min | 其他 | 作业日志被截断，未显示实际测试结果，仅看到上传artifact时无失败文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32551251825/job/96978314845) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法判断测试是否通过或失败。仅看到上传diffusion-failures目录时提示无文件，说明可能没有失败用例，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32551251825/job/96978314845


## [Run #32551034018](https://github.com/sgl-project/sglang/actions/runs/32551034018)
- **分支**: `perf/gdn-strided-target-verify`
- **总耗时**: 197.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32551034018

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726490) |
| base-b-test-16-npu-a3 / run (0) | 36.4min | 环境问题 | Runner 收到关闭信号导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726527) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 3.6min | 环境问题 | Runner收到关闭信号导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96993098879) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.9min | 环境问题 | Runner 在下载 triton-ascend 依赖时收到关闭信号，作业被中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96993834319) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.1min | 其他 | 作业被健康检查快速失败机制跳过，非自身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96999181056) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726490

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段因 runner 收到 shutdown 信号被取消，可能是自托管 runner 被手动停止或服务异常，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726527

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业启动后不久，runner收到shutdown信号（exit code 130），可能是自托管runner被手动停止或服务终止，导致测试未实际运行即失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96993098879

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示在安装 triton-ascend 包（188.5 MB）下载过程中，runner 收到 shutdown 信号，可能是自托管 runner 被手动取消或服务停止，导致容器执行失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96993834319

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业因其他根因作业（base-b-test-16-npu-a3、base-c-test-perf-4/16-npu-a3）失败而被fast-fail跳过，未实际执行测试，属于CI依赖链导致的间接失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96999181056

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726518) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726543) |
| base-b-test-4-npu-a3 / run (0) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726547) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726559) |
| base-a-test-1-npu-a2 / run (0) | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726585) |
| base-b-test-4-npu-a3 / run (1) | 14.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726597) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726675) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726740) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726752) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96977726816) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32551034018/job/96990368672) |


## [Run #32550981238](https://github.com/sgl-project/sglang/actions/runs/32550981238)
- **分支**: `codex/diffusion-t5-bnb-text-encoder`
- **总耗时**: 58.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32550981238

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 44.5min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550981238/job/96977573246) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示GitHub Actions环境准备、Node.js弃用警告及上传失败（无文件）。无法判断测试是否失败或原因，可能因日志截断或作业被提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550981238/job/96977573246


## [Run #32550595701](https://github.com/sgl-project/sglang/actions/runs/32550595701)
- **分支**: `main`
- **总耗时**: 32.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32550595701

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550595701/job/96976658772) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试执行或失败的具体错误信息，仅显示Node.js版本弃用警告和上传失败产物（无文件）。可能因日志截断或作业在测试前被中断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550595701/job/96976658772


## [Run #32550200986](https://github.com/sgl-project/sglang/actions/runs/32550200986)
- **分支**: `amd_helios`
- **总耗时**: 180.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32550200986

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传工件时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629068) |
| base-b-test-16-npu-a3 / run (0) | 46.7min | 代码错误 | NPU PD分离测试失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629107) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 75.6min | 精度回归 | Qwen3.5-9B GSM8K 精度测试失败，0/3 用例全部未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629318) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 31.2min | 环境问题 | Runner收到关闭信号导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96990629735) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业仍标记为失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629068

- **base-b-test-16-npu-a3 / run (0)**: test_npu_pd_disaggregation.py测试失败（exit code 1），其余3个测试通过。该测试涉及PD分离功能，可能是代码逻辑或环境配置问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629107

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 test_npu_qwen3_5_9b_bf16_1p_gsm8k.py 运行 4292 秒后退出码为 1，所有 3 个精度用例均失败，可能因模型精度不达标或环境问题导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629318

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 作业在加载模型分片时，runner收到shutdown信号，可能是自托管runner被手动取消或服务停止，导致容器执行失败，属于环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96990629735

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629092) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629124) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629142) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629169) |
| base-b-test-4-npu-a3 / run (0) | 29.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629211) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629240) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629313) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629328) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96975629344) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96988024035) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550200986/job/96991667562) |


## [Run #32550113139](https://github.com/sgl-project/sglang/actions/runs/32550113139)
- **分支**: `main`
- **总耗时**: 10.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32550113139

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.2min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975429962) |
| base-b-test-4-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430000) |
| base-b-test-16-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430006) |
| base-b-test-4-npu-a3 / run (1) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430042) |
| base-b-test-2-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430072) |
| base-b-test-1-npu-a3 / run (0) | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430092) |
| base-b-test-8-npu-a3 / run (0) | 9.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430107) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430130) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430151) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430158) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430162) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅提示diffusion-failures目录无文件，可能测试未运行或全部通过，但作业被标记为失败，需进一步检查上游步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975429962

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430000

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430006

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430042

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是文件被删除、路径错误或上传失败，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430072

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430092

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或资源被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430107

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430130

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430151

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430158

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430162

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32550113139/job/96975430033) |


## [Run #32549394976](https://github.com/sgl-project/sglang/actions/runs/32549394976)
- **分支**: `lsy`
- **总耗时**: 71.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32549394976

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 60.9min | 环境问题 | GitHub Actions 下载 action 时网络请求失败，导致作业无法正常启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613634) |
| base-b-test-1-npu-a3 / run (0) | 69.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613666) |
| base-b-test-4-npu-a3 / run (0) | 69.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613699) |
| base-b-test-8-npu-a3 / run (0) | 69.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613704) |
| base-b-test-16-npu-a3 / run (0) | 69.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613737) |
| base-b-test-2-npu-a3 / run (0) | 69.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613751) |
| base-b-test-4-npu-a3 / run (1) | 69.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613758) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 69.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613870) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 69.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613935) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 69.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613985) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 69.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973614010) |

- **multimodal-gen-test-1-npu-a3**: 日志显示在准备 action 时出现 'Failed to resolve action download info. Error: An error occurred while sending the request.'，随后重试成功，但整体作业因网络波动或基础设施问题未能完成测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613634

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613666

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613699

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613704

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613737

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、资源被清理或配置问题，属于环境依赖故障。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613751

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613758

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613870

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613935

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613985

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973614010

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 4.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549394976/job/96973613755) |


## [Run #32549339861](https://github.com/sgl-project/sglang/actions/runs/32549339861)
- **分支**: `feat/aiter-gfx950-gdn-decode-backend`
- **总耗时**: 188.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32549339861

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409449) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 3.9min | 环境问题 | Runner 收到关闭信号，作业被取消，非测试本身失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96992773792) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤。可能因日志截断或作业在测试前已终止，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409449

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示 runner 在安装依赖后收到 shutdown 信号，进程以 exit code 130 退出，属于自托管 runner 被外部终止或取消，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96992773792

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409513) |
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409542) |
| base-b-test-4-npu-a3 / run (0) | 28.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409550) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409563) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409581) |
| base-b-test-8-npu-a3 / run (0) | 7.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409613) |
| base-b-test-16-npu-a3 / run (0) | 55.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409623) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409790) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409867) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409916) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96973409917) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96984107988) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 36.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96987114375) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32549339861/job/96987803031) |


## [Run #32548975515](https://github.com/sgl-project/sglang/actions/runs/32548975515)
- **分支**: `dcp-trtllm-mla-decode`
- **总耗时**: 95.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32548975515

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212362) |
| base-b-test-2-npu-a3 / run (0) | 7.8min | 环境问题 | 自定义容器执行失败，NPU测试环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212388) |
| base-b-test-1-npu-a3 / run (0) | 10.1min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212429) |
| base-b-test-4-npu-a3 / run (0) | 11.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212431) |
| base-b-test-4-npu-a3 / run (1) | 10.3min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212436) |
| base-b-test-16-npu-a3 / run (0) | 95.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212445) |
| base-b-test-8-npu-a3 / run (0) | 95.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212478) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 95.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212591) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 95.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212635) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 95.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212644) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 95.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212667) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或未产生失败文件，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212362

- **base-b-test-2-npu-a3 / run (0)**: 日志显示TokenizerManager初始化后，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU测试环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212388

- **base-b-test-1-npu-a3 / run (0)**: 日志显示在CUDA图编译过程中，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212429

- **base-b-test-4-npu-a3 / run (0)**: 日志显示模型加载完成后，在获取ASCEND_OPP_PATH环境变量时，自定义容器实现执行失败，提示联系self-hosted runner管理员，属于NPU运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212431

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动NPU容器时，TokenizerManager初始化后报错"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于NPU容器环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212436

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212445

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212478

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212591

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212635

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是上传失败、过期或被误删，需检查相关存储配置和文件路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212644

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212667

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548975515/job/96983212430) |


## [Run #32548755133](https://github.com/sgl-project/sglang/actions/runs/32548755133)
- **分支**: `zqx/hicache-swa-translate-sentinel-clamp`
- **总耗时**: 167.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32548755133

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 64.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981243) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 66.0min | 精度回归 | Qwen3.5-9B GSM8K 精度测试失败，0/3 通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981559) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 29.0min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96985758446) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未出现测试执行或失败断言。可能因日志截断或作业在测试前被取消，需查看完整日志定位真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981243

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试 test_npu_qwen3_5_9b_bf16_1p_gsm8k.py 返回退出码 1，耗时 3763 秒，所有 3 个测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981559

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示模型权重加载到48%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于运行环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96985758446

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 52.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981275) |
| base-b-test-1-npu-a3 / run (0) | 20.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981305) |
| base-b-test-4-npu-a3 / run (0) | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981306) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981332) |
| base-b-test-2-npu-a3 / run (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981369) |
| base-a-test-1-npu-a2 / run (0) | 6.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981377) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981433) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981503) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981540) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96971981543) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96983431686) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548755133/job/96986942498) |


## [Run #32548707270](https://github.com/sgl-project/sglang/actions/runs/32548707270)
- **分支**: `main`
- **总耗时**: 31.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32548707270

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 29.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971865988) |
| base-b-test-16-npu-a3 / run (0) | 29.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866011) |
| base-b-test-4-npu-a3 / run (0) | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866014) |
| base-b-test-4-npu-a3 / run (1) | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866016) |
| multimodal-gen-test-1-npu-a3 | 29.7min | 其他 | 作业日志不完整，未显示测试执行结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866019) |
| base-b-test-8-npu-a3 / run (0) | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866050) |
| base-b-test-1-npu-a3 / run (0) | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866095) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 29.9min | 环境问题 | CI 依赖的 Azure Blob 存储文件不存在，导致作业无法启动。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866209) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866230) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866255) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 29.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866324) |

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未正确上传，或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971865988

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该文件不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866011

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866014

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866016

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试命令或失败断言，仅有Node.js版本警告和artifact上传提示（无文件）。可能因日志截断或作业在测试前被取消，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866019

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866050

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、数据集或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误。建议检查相关 blob 的路径和存在性。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866095

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明作业尝试下载的构建产物或依赖文件在指定存储位置缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866209

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866230

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866255

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971866324

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548707270/job/96971865990) |


## [Run #32548700954](https://github.com/sgl-project/sglang/actions/runs/32548700954)
- **分支**: `feature/load-reporter`
- **总耗时**: 135.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32548700954

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.5min | 其他 | 作业未执行实际测试，仅上传失败产物但无文件，日志无测试失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796185) |
| base-b-test-8-npu-a3 / run (0) | 4.1min | 代码错误 | NPU测试用例test_npu_eplb_min_rebalancing_utilization_threshold.py执行失败，退出码1。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796367) |
| base-b-test-1-npu-a3 / run (0) | 4.0min | 环境问题 | NPU测试用例test_npu_hicache_mha.py执行失败，返回退出码1，导致整个作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796384) |
| base-b-test-4-npu-a3 / run (1) | 4.9min | 代码错误 | NPU测试用例test_npu_llada2_mini.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796403) |
| base-b-test-2-npu-a3 / run (0) | 4.1min | 代码错误 | NPU专家并行测试文件执行失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796412) |
| base-b-test-16-npu-a3 / run (0) | 36.0min | 代码错误 | NPU测试test_npu_deepep.py执行失败，退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796427) |
| base-b-test-4-npu-a3 / run (0) | 4.4min | 代码错误 | HiCache MLA 测试失败，返回退出码1 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796432) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.4min | 精度回归 | NPU精度测试用例失败，导致作业整体失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796574) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.2min | 精度回归 | NPU精度测试失败，qwen3_vl_8b_bf16_2p_gsm8k测试用例未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796577) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.2min | 精度回归 | NPU精度测试用例moonshotai_moonlight_16b_a3b_bf16_1p_gsm8k失败，0/3通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796621) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 4.4min | 性能回归 | NPU性能测试未达预期，测试用例执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96982880713) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段后直接进入上传diffusion-failures产物步骤，但未找到任何文件，未出现测试执行或失败断言，可能因前置条件未满足或作业被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796185

- **base-b-test-8-npu-a3 / run (0)**: 该测试文件在44秒内失败，返回退出码1，导致整个作业终止。可能是测试逻辑或断言错误，需查看具体测试输出定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796367

- **base-b-test-1-npu-a3 / run (0)**: 测试文件test/registered/npu/basic_function/HiCache/test_npu_hicache_mha.py在运行43秒后失败，退出码为1。日志中未显示具体错误信息，但测试摘要显示0/11通过，可能是环境配置或依赖问题导致测试无法正常运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796384

- **base-b-test-4-npu-a3 / run (1)**: 测试文件test/registered/npu/basic_function/dllm/test_npu_llada2_mini.py在运行43秒后返回退出码1，导致整个作业失败。具体失败原因需查看该测试用例的详细输出，可能是模型加载、推理或断言错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796403

- **base-b-test-2-npu-a3 / run (0)**: 测试文件test_npu_expert_distribution_recorder_mode.py在运行约44秒后失败，0/6测试通过，具体错误信息未在日志中显示，可能涉及代码逻辑或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796412

- **base-b-test-16-npu-a3 / run (0)**: 在expert_parallelism测试中，test_npu_deepep.py运行43秒后失败（exit code 1），而其他两个测试通过。可能是该测试用例存在代码逻辑错误或环境配置问题，导致测试脚本异常退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796427

- **base-b-test-4-npu-a3 / run (0)**: 测试文件 test_npu_hicache_mla.py 执行失败，退出码为1，测试摘要显示0/5通过，具体错误信息未在日志中展示，可能涉及功能实现或环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796432

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 测试用例 test_npu_glm5_top64_pruned_bf16_8p_gsm8k.py 返回退出码1，测试摘要显示0/1通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796574

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 测试套件中2个用例全部失败，其中qwen3_vl_8b_bf16_2p_gsm8k.py返回退出码1，耗时约44秒，属于精度回归问题，需检查模型输出与预期是否一致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796577

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试用例test_npu_moonlight_16b_a3b_bf16_1p_gsm8k.py返回退出码1，3个测试全部失败，耗时44秒，属于精度回归问题，可能由模型权重或代码改动导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796621

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试用例test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py在44.62秒内失败，0/1通过，表明性能未达标或执行出错，导致作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96982880713

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796352) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548700954/job/96971796597) |


## [Run #32548618621](https://github.com/sgl-project/sglang/actions/runs/32548618621)
- **分支**: `lsy`
- **总耗时**: 17.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32548618621

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567397) |
| base-b-test-4-npu-a3 / run (1) | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567474) |
| base-b-test-4-npu-a3 / run (0) | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567477) |
| base-b-test-16-npu-a3 / run (0) | 16.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567482) |
| base-b-test-1-npu-a3 / run (0) | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567486) |
| base-b-test-2-npu-a3 / run (0) | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567494) |
| base-b-test-8-npu-a3 / run (0) | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567602) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567701) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567730) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 16.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567783) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 16.4min | 环境问题 | CI作业因Azure Blob存储中指定的blob不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567785) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败断言信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或日志被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567397

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567474

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567477

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567482

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567486

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567494

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567602

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567701

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567730

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到，可能是文件被删除、路径错误或上传未完成，属于外部依赖缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567783

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示BlobNotFound错误，表明作业尝试下载或访问的存储对象缺失，可能是资源未上传或路径错误，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567785

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548618621/job/96971567506) |


## [Run #32548508586](https://github.com/sgl-project/sglang/actions/runs/32548508586)
- **分支**: `dots-note-for-sglang`
- **总耗时**: 183.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32548508586

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249824) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 12.4min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96981851906) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 4.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96991326804) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体错误信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249824

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行544秒后失败，0/1通过，属于性能指标未达到预期要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96981851906

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96991326804

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 20.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249893) |
| base-b-test-16-npu-a3 / run (0) | 60.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249900) |
| base-b-test-1-npu-a3 / run (0) | 25.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249901) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249918) |
| base-b-test-4-npu-a3 / run (1) | 14.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249943) |
| base-b-test-8-npu-a3 / run (0) | 9.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249947) |
| base-b-test-4-npu-a3 / run (0) | 29.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971249949) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971250145) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 89.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971250151) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971250156) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96971250169) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 38.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96982676770) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548508586/job/96985123835) |


## [Run #32548236937](https://github.com/sgl-project/sglang/actions/runs/32548236937)
- **分支**: `feat/mapped-layer-courier`
- **总耗时**: 152.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32548236937

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.5min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549267) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 72.3min | 环境问题 | 自定义容器执行失败，导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549529) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 17.3min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96982661664) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 12.6min | 环境问题 | 自定义容器执行失败，NPU性能测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96983462281) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅有Node 20弃用警告和diffusion-failures目录无文件上传提示，可能为作业提前结束或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549267

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在05:42:38出现错误："Executing the custom container implementation failed"，提示联系自托管runner管理员，属于容器环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549529

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示在初始化分布式训练时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或资源问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96982661664

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示测试运行正常，但在05:42:38时出现"Executing the custom container implementation failed"错误，导致作业中断。这属于自托管runner环境问题，而非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96983462281

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-2-npu-a3 / run (0) | 19.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549378) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549380) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549386) |
| base-b-test-16-npu-a3 / run (0) | 51.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549396) |
| base-b-test-1-npu-a3 / run (0) | 23.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549398) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549400) |
| base-b-test-4-npu-a3 / run (1) | 14.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549409) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549491) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549542) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96970549611) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32548236937/job/96980030111) |


## [Run #32547515476](https://github.com/sgl-project/sglang/actions/runs/32547515476)
- **分支**: `dcp-trtllm-mla-decode`
- **总耗时**: 32.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32547515476

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 28.7min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620055) |
| base-b-test-1-npu-a3 / run (0) | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620070) |
| base-b-test-4-npu-a3 / run (1) | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620076) |
| base-b-test-2-npu-a3 / run (0) | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620079) |
| base-b-test-16-npu-a3 / run (0) | 31.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620085) |
| base-b-test-8-npu-a3 / run (0) | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620089) |
| base-b-test-4-npu-a3 / run (0) | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620155) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | 代码错误 | 测试文件缺少主入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620159) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620305) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620393) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620475) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 31.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620477) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含实际测试执行和失败信息，仅显示上传diffusion-failures工件时未找到文件，说明测试可能未产生失败样本或提前退出，需查看完整日志定位问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620055

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620070

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620076

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620079

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620085

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/环境配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620089

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620155

- **base-a-test-1-npu-a2 / run (0)**: test/registered/dcp/test_trtllm_mla_dcp_decode_hooks.py 缺少 `if __name__ == "__main__":` 入口，导致 pytest 风格测试在 `python3 file.py -f` 下静默跳过，collect_tests 抛出 ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620159

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620305

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620393

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620475

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32547515476/job/96968620477


## [Run #32546336331](https://github.com/sgl-project/sglang/actions/runs/32546336331)
- **分支**: `main`
- **总耗时**: 52.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32546336331

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 35.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433740) |
| base-b-test-8-npu-a3 / run (0) | 51.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433789) |
| base-b-test-2-npu-a3 / run (0) | 51.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433791) |
| base-b-test-1-npu-a3 / run (0) | 51.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433796) |
| base-b-test-4-npu-a3 / run (1) | 51.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433859) |
| base-b-test-16-npu-a3 / run (0) | 51.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433876) |
| base-b-test-4-npu-a3 / run (0) | 51.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433903) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 51.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965434085) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 51.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965434089) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 51.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965434132) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 51.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965434133) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业正常执行了checkout和upload-artifact，但未包含测试运行的具体输出或错误信息。上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但无法从现有日志判断作业失败的具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433740

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433789

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失或配置错误，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433791

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433796

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433859

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433876

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、资源被清理或配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433903

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965434085

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965434089

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965434132

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965434133

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32546336331/job/96965433755) |


## [Run #32545921948](https://github.com/sgl-project/sglang/actions/runs/32545921948)
- **分支**: `main`
- **总耗时**: 9.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32545921948

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262182) |
| base-b-test-16-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262220) |
| base-b-test-4-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262254) |
| base-b-test-1-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262266) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262300) |
| base-b-test-4-npu-a3 / run (1) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262302) |
| base-b-test-2-npu-a3 / run (0) | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262313) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262371) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262373) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262377) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262425) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未展示multimodal-gen测试的具体执行和失败输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262182

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262220

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262254

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262266

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262300

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262302

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262313

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262371

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262373

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262377

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262425

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545921948/job/96964262272) |


## [Run #32545378215](https://github.com/sgl-project/sglang/actions/runs/32545378215)
- **分支**: `perf/w4a8-ll-requant-tuning`
- **总耗时**: 275.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32545378215

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.9min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837090) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 106.8min | 超时 | 性能测试服务启动超时，服务器未在60秒内就绪。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96979645180) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 62.3min | 环境问题 | Runner 收到关闭信号导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96986323695) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出，仅显示上传diffusion-failures目录时未找到文件，无法判断测试是否通过或失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837090

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: bench_serving等待http://127.0.0.1:20666/v1/models就绪超时，服务器未启动成功，随后runner收到关闭信号，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96979645180

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 作业运行约1小时后，runner 收到 shutdown 信号（exit code 130），可能是自托管 runner 被手动停止或服务终止，并非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96986323695

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837140) |
| base-b-test-2-npu-a3 / run (0) | 19.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837156) |
| base-a-test-1-npu-a2 / run (0) | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837172) |
| base-b-test-4-npu-a3 / run (0) | 30.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837193) |
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837200) |
| base-b-test-16-npu-a3 / run (0) | 53.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837201) |
| base-b-test-8-npu-a3 / run (0) | 7.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837233) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837251) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837266) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837276) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96962837358) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96977880302) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 20.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545378215/job/96981731346) |


## [Run #32545220119](https://github.com/sgl-project/sglang/actions/runs/32545220119)
- **分支**: `feat/mapped-weight-readahead`
- **总耗时**: 68.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32545220119

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545220119/job/96962411292) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或未产生失败文件，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545220119/job/96962411292


## [Run #32545207473](https://github.com/sgl-project/sglang/actions/runs/32545207473)
- **分支**: `deepseek_v4_dspark_suppport_pp_pd`
- **总耗时**: 279.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32545207473

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.8min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383146) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 64.3min | AI调用失败 | HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96985029811) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含实际测试执行和失败信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383146

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out.
  链接: https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96985029811

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (1) | 14.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383188) |
| base-b-test-16-npu-a3 / run (0) | 52.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383193) |
| base-b-test-4-npu-a3 / run (0) | 30.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383198) |
| base-b-test-8-npu-a3 / run (0) | 7.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383206) |
| base-b-test-1-npu-a3 / run (0) | 21.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383217) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 43.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383265) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383281) |
| base-b-test-2-npu-a3 / run (0) | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383308) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383333) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383357) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96962383361) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96976402634) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96979300008) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32545207473/job/96981370649) |


## [Run #32545065555](https://github.com/sgl-project/sglang/actions/runs/32545065555)
- **分支**: `fix/mapped-weight-metadata`
- **总耗时**: 67.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32545065555

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.5min | AI调用失败 | HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out. | [job link](https://github.com/sgl-project/sglang/actions/runs/32545065555/job/96962002451) |

- **multimodal-gen-test-1-npu-a3**: HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out.
  链接: https://github.com/sgl-project/sglang/actions/runs/32545065555/job/96962002451


## [Run #32544682431](https://github.com/sgl-project/sglang/actions/runs/32544682431)
- **分支**: `yuychang/k3-kda-inproj-fusion`
- **总耗时**: 13.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32544682431

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026747) |
| base-b-test-8-npu-a3 / run (0) | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026864) |
| base-b-test-4-npu-a3 / run (1) | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026865) |
| base-b-test-4-npu-a3 / run (0) | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026887) |
| base-b-test-1-npu-a3 / run (0) | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026938) |
| base-b-test-16-npu-a3 / run (0) | 12.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026949) |
| base-b-test-2-npu-a3 / run (0) | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026969) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961027066) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961027094) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961027126) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961027130) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示runner启动、依赖下载和上传工件（无文件）。可能因日志截断或作业在测试前被取消，无法判断真实失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026747

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026864

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026865

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026887

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是缓存或依赖文件未正确上传，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026938

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026949

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026969

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径错误，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961027066

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是上传失败、过期或被误删，需检查相关存储配置和文件路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961027094

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961027126

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径配置错误，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961027130

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32544682431/job/96961026875) |


## [Run #32544641005](https://github.com/sgl-project/sglang/actions/runs/32544641005)
- **分支**: `main`
- **总耗时**: 27.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32544641005

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 23.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926026) |
| base-b-test-4-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926120) |
| base-b-test-1-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926136) |
| base-b-test-16-npu-a3 / run (0) | 26.9min | 环境问题 | 日志显示Azure Blob存储中的文件不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926138) |
| base-b-test-8-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926147) |
| base-b-test-4-npu-a3 / run (1) | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926202) |
| base-b-test-2-npu-a3 / run (0) | 26.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926214) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926261) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926277) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926282) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 26.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926306) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅有GitHub Actions运行环境准备、Node版本警告及上传artifact时未找到文件的提示，无法判断测试失败的具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926026

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重、缓存或构建产物）已被删除或路径错误，需检查相关资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926120

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926136

- **base-b-test-16-npu-a3 / run (0)**: 作业在下载或访问某个Blob文件时，服务器返回BlobNotFound错误，说明该文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926138

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926147

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926202

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926214

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926261

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926277

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926282

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926306

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32544641005/job/96960926105) |


## [Run #32544377288](https://github.com/sgl-project/sglang/actions/runs/32544377288)
- **分支**: `feat/layerwise-all-literal`
- **总耗时**: 73.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32544377288

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 64.3min | 其他 | 作业日志被截断，未显示实际失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544377288/job/96960215777) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。最后仅显示上传diffusion-failures目录时未找到文件，可能测试未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544377288/job/96960215777


## [Run #32544290089](https://github.com/sgl-project/sglang/actions/runs/32544290089)
- **分支**: `main`
- **总耗时**: 7.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32544290089

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005372) |
| base-b-test-1-npu-a3 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005395) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005411) |
| base-b-test-2-npu-a3 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005463) |
| base-b-test-16-npu-a3 / run (0) | 7.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005470) |
| base-b-test-4-npu-a3 / run (1) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005487) |
| base-b-test-4-npu-a3 / run (0) | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005491) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005591) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005593) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005679) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005769) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或模型权重在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005372

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储对象缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005395

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005411

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005463

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、资源被清理或配置问题，属于环境或基础设施故障，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005470

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005487

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005491

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005591

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005593

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005679

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005769

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32544290089/job/96960005444) |


## [Run #32544255126](https://github.com/sgl-project/sglang/actions/runs/32544255126)
- **分支**: `feat/mapped-layer-courier`
- **总耗时**: 74.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32544255126

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 65.2min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传空产物信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544255126/job/96959907457) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅有Node 20弃用警告和diffusion-failures目录无文件上传的提示，可能为作业提前结束或测试未执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544255126/job/96959907457


## [Run #32544030553](https://github.com/sgl-project/sglang/actions/runs/32544030553)
- **分支**: `codex/diffusion-component-hub-subfolders`
- **总耗时**: 80.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32544030553

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 64.7min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32544030553/job/96959300715) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions的初始化、上传artifact（未找到文件）和清理步骤，没有测试执行或失败的具体输出，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32544030553/job/96959300715


## [Run #32543977890](https://github.com/sgl-project/sglang/actions/runs/32543977890)
- **分支**: `main`
- **总耗时**: 6.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32543977890

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543977890/job/96959159246) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543977890/job/96959159246


## [Run #32543681823](https://github.com/sgl-project/sglang/actions/runs/32543681823)
- **分支**: `lsy`
- **总耗时**: 98.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32543681823

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.3min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505625) |
| base-b-test-1-npu-a3 / run (0) | 98.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505634) |
| base-b-test-2-npu-a3 / run (0) | 98.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505638) |
| base-b-test-4-npu-a3 / run (1) | 98.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505645) |
| base-b-test-16-npu-a3 / run (0) | 98.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505674) |
| base-b-test-4-npu-a3 / run (0) | 98.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505691) |
| base-b-test-8-npu-a3 / run (0) | 98.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505756) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 98.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505834) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 98.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505884) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 98.0min | 环境问题 | CI作业因Azure Blob存储中找不到指定文件而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505931) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 98.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505987) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，仅显示上传diffusion-failures目录时未找到文件，无法判断具体失败原因，可能为测试未生成失败产物或日志不完整。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505625

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505634

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505638

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或资源可用性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505645

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505674

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505691

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505756

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505834

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505884

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示BlobNotFound错误，说明作业依赖的某个文件（如模型权重或测试数据）在存储中不存在，可能是上传缺失或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505931

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505987

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543681823/job/96959505663) |


## [Run #32543451029](https://github.com/sgl-project/sglang/actions/runs/32543451029)
- **分支**: `feat/diffusion-maybe-record-function`
- **总耗时**: 72.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32543451029

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543451029/job/96974524500) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，仅显示上传diffusion-failures目录时无文件，未包含测试执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543451029/job/96974524500


## [Run #32543423862](https://github.com/sgl-project/sglang/actions/runs/32543423862)
- **分支**: `wca-rebased`
- **总耗时**: 207.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32543423862

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669631) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 81.7min | 环境问题 | 自定义容器执行失败，NPU测试中途异常终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669927) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 41.5min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96975959691) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告、上传artifact（无文件）及清理步骤，未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669631

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在04:53:01时出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669927

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示性能测试运行正常，但容器执行中途报错“Executing the custom container implementation failed”，属于runner或容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96975959691

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 51.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669700) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669733) |
| base-b-test-4-npu-a3 / run (0) | 29.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669736) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669737) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669759) |
| base-b-test-1-npu-a3 / run (0) | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669776) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669805) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 28.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669890) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957669910) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96957670039) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96973688054) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543423862/job/96977216878) |


## [Run #32543410873](https://github.com/sgl-project/sglang/actions/runs/32543410873)
- **分支**: `amd/dsv4-aiter-fused-mhc-cross-layer`
- **总耗时**: 287.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32543410873

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640210) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 67.2min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96981950856) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示Node.js版本弃用警告和上传artifact时未找到文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640210

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示性能测试正常运行中，但突然报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96981950856

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640245) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640282) |
| base-b-test-2-npu-a3 / run (0) | 20.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640284) |
| base-b-test-8-npu-a3 / run (0) | 7.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640323) |
| base-b-test-4-npu-a3 / run (1) | 16.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640326) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640340) |
| base-b-test-16-npu-a3 / run (0) | 46.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640404) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640444) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640449) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640450) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 36.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96957640496) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96973215825) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 36.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96976304686) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543410873/job/96976840535) |


## [Run #32543329727](https://github.com/sgl-project/sglang/actions/runs/32543329727)
- **分支**: `mlx-032-torch-213-bridge`
- **总耗时**: 27.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32543329727

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 17.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373170) |
| base-b-test-4-npu-a3 / run (1) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373201) |
| base-b-test-16-npu-a3 / run (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373204) |
| base-b-test-4-npu-a3 / run (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373210) |
| base-b-test-8-npu-a3 / run (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373243) |
| base-b-test-1-npu-a3 / run (0) | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373253) |
| base-b-test-2-npu-a3 / run (0) | 26.8min | 环境问题 | CI作业因Azure Blob存储中找不到指定文件而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373274) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373296) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373318) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373364) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 26.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373432) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或未产生失败文件，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373170

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373201

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传失败、路径错误或存储被清理所致，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373204

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或配置问题，需检查相关存储路径或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373210

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/环境配置问题，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373243

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373253

- **base-b-test-2-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业依赖的某个blob文件不存在或已被删除，可能是构建产物或缓存缺失，属于环境配置或资源清理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373274

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373296

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373318

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373364

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373432

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32543329727/job/96957373233) |


## [Run #32542984477](https://github.com/sgl-project/sglang/actions/runs/32542984477)
- **分支**: `hicache-buffer-anchor-lock-oss`
- **总耗时**: 123.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32542984477

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.8min | 其他 | 作业未显示实际测试失败，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444282) |
| base-b-test-16-npu-a3 / run (0) | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444388) |
| base-b-test-2-npu-a3 / run (0) | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444405) |
| base-b-test-1-npu-a3 / run (0) | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444428) |
| base-b-test-4-npu-a3 / run (1) | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444459) |
| base-b-test-8-npu-a3 / run (0) | 123.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444463) |
| base-b-test-4-npu-a3 / run (0) | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444495) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444623) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444633) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444701) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 123.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444703) |

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures目录时无文件，可能测试未运行或全部通过，但作业被标记失败，需检查前置步骤是否被跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444282

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444388

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444405

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重、缓存或构建产物）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444428

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444459

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444463

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444495

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444623

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444633

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444701

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444703

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542984477/job/96956444468) |


## [Run #32542789457](https://github.com/sgl-project/sglang/actions/runs/32542789457)
- **分支**: `main`
- **总耗时**: 18.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32542789457

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.7min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955920958) |
| base-b-test-8-npu-a3 / run (0) | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921033) |
| base-b-test-1-npu-a3 / run (0) | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921051) |
| base-b-test-4-npu-a3 / run (1) | 17.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921088) |
| base-b-test-2-npu-a3 / run (0) | 17.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921092) |
| base-b-test-16-npu-a3 / run (0) | 17.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921153) |
| base-b-test-4-npu-a3 / run (0) | 17.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921179) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921332) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 17.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921345) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921355) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 17.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921356) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示GitHub Actions环境准备、Node.js弃用警告及上传失败文件（无文件）等常规信息，无法判断测试失败的具体原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955920958

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921033

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921051

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921088

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921092

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921153

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，属于环境/资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921179

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921332

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能因资源清理或配置变更引起，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921345

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921355

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921356

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542789457/job/96955921027) |


## [Run #32542372361](https://github.com/sgl-project/sglang/actions/runs/32542372361)
- **分支**: `qwen-qiaolin`
- **总耗时**: 262.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32542372361

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755444) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 27.7min | 性能回归 | NPU性能测试用例失败，Qwen3-235B模型8卡测试未通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96972605428) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96982799245) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未展示测试执行过程或错误信息，无法判断具体失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755444

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_qwen3_235b_w8a8_8p_in3k5_out1k5_50ms.py返回退出码1，耗时1326秒，3个测试全部失败，疑似性能未达预期或运行错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96972605428

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查检测到base-c-test-perf-16-npu-a3作业失败，被判定为根因作业，因此本作业（base-c-test-perf-2-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96982799245

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-16-npu-a3 / run (0) | 49.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755541) |
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755543) |
| base-b-test-4-npu-a3 / run (1) | 15.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755548) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755624) |
| base-b-test-1-npu-a3 / run (0) | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755627) |
| base-b-test-4-npu-a3 / run (0) | 31.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755629) |
| base-a-test-1-npu-a2 / run (0) | 11.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755650) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755761) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755804) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 113.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954755806) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96954756173) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96971739997) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 23.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32542372361/job/96975133657) |


## [Run #32541519730](https://github.com/sgl-project/sglang/actions/runs/32541519730)
- **分支**: `cheng/gc-cut-1`
- **总耗时**: 354.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32541519730

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417198) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 72.5min | 环境问题 | Runner收到关闭信号导致作业中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96984053190) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法定位具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417198

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示作业在正常运行中，但runner突然收到shutdown信号，可能是自托管runner被停止或取消，导致容器执行失败，并非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96984053190

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417264) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417290) |
| base-b-test-2-npu-a3 / run (0) | 21.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417297) |
| base-b-test-16-npu-a3 / run (0) | 76.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417315) |
| base-b-test-4-npu-a3 / run (0) | 31.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417346) |
| base-b-test-8-npu-a3 / run (0) | 8.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417347) |
| base-b-test-1-npu-a3 / run (0) | 24.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417353) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417533) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 34.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417536) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 125.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417610) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96952417690) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 17.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96969790933) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 40.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96973169638) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 21.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32541519730/job/96974402870) |


---
*Auto-generated by npu_pr_monitor.py*