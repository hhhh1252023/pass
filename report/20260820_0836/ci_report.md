# NPU CI 执行监控
**生成时间**: 2026-08-20 00:36 UTC
**分析 Run 数**: 46

---

## 📊 本次执行总结

- **成功 Job 数**: 27
- **失败 Run 数**: 46
- **成功 Job 平均耗时**: 4.9min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-a-test-1-npu-a2 / run (0) | 14.2min | #32265599205 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540596) |
| base-a-test-1-npu-a2 / run (0) | 10.0min | #32272583378 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459783) |
| base-a-test-1-npu-a2 / run (0) | 6.6min | #32302865983 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384706) |
| base-a-test-1-npu-a2 / run (0) | 6.3min | #32274443191 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770708) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | #32302940992 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893280) |
| base-a-test-1-npu-a2 / run (0) | 6.0min | #32272089424 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884881) |
| base-a-test-1-npu-a2 / run (0) | 5.9min | #32266295648 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111662926) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | #32273609126 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838267) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | #32286037968 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839814) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | #32266887725 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613736) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| base-a-test-1-npu-a2 / run (0) | 37 | 21 | 0 | 15 | 1 |
| multimodal-gen-test-1-npu-a3 | 46 | 0 | 15 | 0 | 31 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32273609126<br>[#35337 [XPU][CI] key persistent JIT kernel cache by image content ID](https://github.com/sgl-project/sglang/pull/35337) | `xpu-kernel-cache-image-id` | 235.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32273609126) |
| #32271980800<br>[#33554 Add new spec-dec support and quant recipe for Nano v3](https://github.com/sgl-project/sglang/pull/33554) | `nemotron-3.5-spec-comparison` | 204.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32271980800) |
| #32268787864<br>[#35077 [Fix] Support Kimi-K3 ModelOpt mixed NVFP4/FP8 checkpoint](https://github.com/sgl-project/sglang/pull/35077) | `main` | 164.5min | 11 | base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32268787864) |
| #32304721645<br>[#27010 [HiCache] Fix PP inconsistency with HiCache L3 (#22607)](https://github.com/sgl-project/sglang/pull/27010) | `sglang_pp_bug4` | 131.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32304721645) |
| #32265599205<br>[#27770 [P/D disagg] Decode-side radix cache for SWA hybrid models (unified radix tree)](https://github.com/sgl-project/sglang/pull/27770) | `idhanani/unified-radix-swa-fix` | 119.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32265599205) |
| #32274443191<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 102.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32274443191) |
| #32266199885<br>[#35538 [diffusion] fix: stop reserving NCCL device buffers for single-rank groups](https://github.com/sgl-project/sglang/pull/35538) | `fix/single-rank-nccl-vram` | 82.7min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32266199885) |
| #32294556347<br>[#34893 [Diffusion] Add MiniMax H3 cube sparse attention](https://github.com/sgl-project/sglang/pull/34893) | `codex/minimax-h3-cube-sparse-attn` | 79.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32294556347) |
| #32274079488<br>[#35538 [diffusion] fix: stop reserving NCCL device buffers for single-rank groups](https://github.com/sgl-project/sglang/pull/35538) | `fix/single-rank-nccl-vram` | 75.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32274079488) |
| #32265131842<br>[#33554 Add new spec-dec support and quant recipe for Nano v3](https://github.com/sgl-project/sglang/pull/33554) | `nemotron-3.5-spec-comparison` | 73.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32265131842) |
| #32305203302<br>[#34955 [Diffusion] Align MiniMax H3 request RNG with reference](https://github.com/sgl-project/sglang/pull/34955) | `codex/minimax-h3-reference-rng` | 71.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32305203302) |
| #32282822097<br>[#35511 [diffusion] CI: add minimax-h3 ref2va audio consistency coverage and guard peak vram](https://github.com/sgl-project/sglang/pull/35511) | `codex/minimax-h3-ref2va-audio-ci` | 68.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32282822097) |
| #32305460556<br>[#33036 [diffusion] feat: overlap declared-parallel pipeline stages](https://github.com/sgl-project/sglang/pull/33036) | `feat/parallel-stage-groups` | 67.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32305460556) |
| #32291208807<br>[#35538 [diffusion] fix: stop reserving NCCL device buffers for single-rank groups](https://github.com/sgl-project/sglang/pull/35538) | `fix/single-rank-nccl-vram` | 67.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32291208807) |
| #32304639300<br>[#35511 [diffusion] CI: add minimax-h3 ref2va audio consistency coverage and guard peak vram](https://github.com/sgl-project/sglang/pull/35511) | `codex/minimax-h3-ref2va-audio-ci` | 66.8min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32304639300) |
| #32296057568<br>[#32710 [RFC] Rust Tree Core Full Component](https://github.com/sgl-project/sglang/pull/32710) | `jialino/rust-tree-core-full` | 58.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32296057568) |
| #32298749607<br>[#35401 [Fix] Write the req_to_token page tail so rows stay valid over whole pages](https://github.com/sgl-project/sglang/pull/35401) | `lsyin/page-tail-write` | 51.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32298749607) |
| #32308517891<br>[#35239 Rainj me/rust server refactor2](https://github.com/sgl-project/sglang/pull/35239) | `rainj-me/rust-server-refactor2` | 51.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32308517891) |
| #32272583378 | `fix/dsa-movekv-page-aware` | 46.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32272583378) |
| #32302865983<br>[#35568 Revert "[Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend"](https://github.com/sgl-project/sglang/pull/35568) | `main` | 45.4min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32302865983) |
| #32280512153<br>[#35546 [EAGLE] Prune draft-extend logits to selected rows](https://github.com/sgl-project/sglang/pull/35546) | `agentx-upstream/eagle-selected-row-logits-20260819` | 43.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32280512153) |
| #32304831260<br>[#34679 fix(constrained): reject NUL bytes in grammar specs to stop an xgrammar segfault](https://github.com/sgl-project/sglang/pull/34679) | `junshen/fix-nul-grammar-segfault` | 43.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32304831260) |
| #32287709453<br>[#35540 [HiCache] Split the host-memory budget across co-located ranks](https://github.com/sgl-project/sglang/pull/35540) | `cctry/hicache/per-host-memory-budget` | 42.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32287709453) |
| #32290707158<br>[#35017 [Scheduler] Add configurable decode interval after prefill](https://github.com/sgl-project/sglang/pull/35017) | `main` | 40.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32290707158) |
| #32286037968<br>[#35545 [Qwen3.5][MTP] Preserve online NVFP4 draft quantization for mixed checkpoints](https://github.com/sgl-project/sglang/pull/35545) | `main` | 40.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32286037968) |
| #32266887725<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 40.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32266887725) |
| #32296985048<br>[#35041 [DSA] Trim top-k v2 output modes and tighten its PDL waits](https://github.com/sgl-project/sglang/pull/35041) | `main` | 33.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32296985048) |
| #32266295648<br>[#35269 [UnifiedTree] feat: support runtime attach/detach](https://github.com/sgl-project/sglang/pull/35269) | `main` | 29.8min | 11 | base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32266295648) |
| #32302940992<br>[#35401 [Fix] Write the req_to_token page tail so rows stay valid over whole pages](https://github.com/sgl-project/sglang/pull/35401) | `lsyin/page-tail-write` | 28.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32302940992) |
| #32301099404<br>[#35409 fix(disagg): allow fake transfer with decode DCP](https://github.com/sgl-project/sglang/pull/35409) | `main` | 24.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32301099404) |
| #32302615903<br>[#35412 [Fix] Land the decode mamba checkpoint depth on the tree page under DCP](https://github.com/sgl-project/sglang/pull/35412) | `kpham/mamba-track-interval-tree-page` | 23.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32302615903) |
| #32294456267<br>[#32440 fix(gemma4): quantize MTP bridge projections](https://github.com/sgl-project/sglang/pull/32440) | `main` | 20.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32294456267) |
| #32306313136<br>[#35239 Rainj me/rust server refactor2](https://github.com/sgl-project/sglang/pull/35239) | `rainj-me/rust-server-refactor2` | 19.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32306313136) |
| #32299572780<br>[#33606 feat(openai): Accept the input_audio content part in chat completions](https://github.com/sgl-project/sglang/pull/33606) | `main` | 17.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32299572780) |
| #32284657305<br>[#34908 Support Intern-S2-Mobius FP8](https://github.com/sgl-project/sglang/pull/34908) | `main` | 14.6min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32284657305) |
| #32308365479<br>[#34679 fix(constrained): reject NUL bytes in grammar specs to stop an xgrammar segfault](https://github.com/sgl-project/sglang/pull/34679) | `junshen/fix-nul-grammar-segfault` | 14.1min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32308365479) |
| #32294029282<br>[#35511 [diffusion] CI: add minimax-h3 ref2va audio consistency coverage and guard peak vram](https://github.com/sgl-project/sglang/pull/35511) | `codex/minimax-h3-ref2va-audio-ci` | 12.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32294029282) |
| #32272089424<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 12.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32272089424) |
| #32271002353<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 11.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32271002353) |
| #32289853679<br>[#29525 [Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend](https://github.com/sgl-project/sglang/pull/29525) | `main` | 8.9min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32289853679) |
| #32296305056<br>[#35540 [HiCache] Split the host-memory budget across co-located ranks](https://github.com/sgl-project/sglang/pull/35540) | `main` | 7.5min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32296305056) |
| #32309195232<br>[#34679 fix(constrained): reject NUL bytes in grammar specs to stop an xgrammar segfault](https://github.com/sgl-project/sglang/pull/34679) | `main` | 7.4min | 11 | base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32309195232) |
| #32273468158<br>[#33569 [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's](https://github.com/sgl-project/sglang/pull/33569) | `minimax-h3-on-npu-support` | 7.2min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32273468158) |
| #32308584934<br>[#35574 [HiCache] Simple style change for buffer mode](https://github.com/sgl-project/sglang/pull/35574) | `main` | 6.2min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32308584934) |
| #32298274193<br>[#35401 [Fix] Write the req_to_token page tail so rows stay valid over whole pages](https://github.com/sgl-project/sglang/pull/35401) | `lsyin/page-tail-write` | 6.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32298274193) |
| #32287282126<br>[#35540 [HiCache] Split the host-memory budget across co-located ranks](https://github.com/sgl-project/sglang/pull/35540) | `cctry/hicache/per-host-memory-budget` | 5.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32287282126) |

---


## [Run #32309195232](https://github.com/sgl-project/sglang/actions/runs/32309195232)
- **分支**: `main`
- **总耗时**: 7.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32309195232

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775125) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | 环境问题 | 测试全部通过但容器执行失败，属于自托管runner环境问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775180) |
| multimodal-gen-test-1-npu-a3 | 3.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775183) |
| base-b-test-8-npu-a3 / run (0) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775209) |
| base-b-test-2-npu-a3 / run (0) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775233) |
| base-b-test-4-npu-a3 / run (0) | 5.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775266) |
| base-b-test-4-npu-a3 / run (1) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775285) |
| base-b-test-16-npu-a3 / run (0) | 5.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775420) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775545) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775555) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775851) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775962) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775125

- **base-a-test-1-npu-a2 / run (0)**: 日志显示2个NPU测试均通过（2/2 passed），但随后出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于容器执行环境故障，非代码或测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775180

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能确认上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本或提前退出，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775183

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775209

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775233

- **base-b-test-4-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775266

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775285

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775420

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775545

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775555

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775851

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32309195232/job/96248775962


## [Run #32308584934](https://github.com/sgl-project/sglang/actions/runs/32308584934)
- **分支**: `main`
- **总耗时**: 6.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32308584934

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690695) |
| base-b-test-1-npu-a3 / run (0) | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690757) |
| base-b-test-2-npu-a3 / run (0) | 5.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690849) |
| base-b-test-8-npu-a3 / run (0) | 5.7min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690857) |
| base-b-test-4-npu-a3 / run (1) | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690875) |
| base-b-test-16-npu-a3 / run (0) | 5.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690966) |
| base-b-test-4-npu-a3 / run (0) | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690977) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246691353) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246691379) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246691413) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246691609) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690695

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690757

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690849

- **base-b-test-8-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明CI作业尝试下载的Azure Blob文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690857

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690875

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690966

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690977

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246691353

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246691379

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246691413

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在 Azure Blob 存储中找到，可能是文件被误删、路径错误或上传失败，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246691609

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32308584934/job/96246690867) |


## [Run #32308517891](https://github.com/sgl-project/sglang/actions/runs/32308517891)
- **分支**: `rainj-me/rust-server-refactor2`
- **总耗时**: 51.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32308517891

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 42.6min | 其他 | 作业未显示明确失败原因，仅上传失败产物时未找到文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545067) |
| base-b-test-8-npu-a3 / run (0) | 50.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545256) |
| base-b-test-2-npu-a3 / run (0) | 50.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545272) |
| base-b-test-4-npu-a3 / run (0) | 50.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545295) |
| base-b-test-16-npu-a3 / run (0) | 50.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545346) |
| base-b-test-1-npu-a3 / run (0) | 50.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545424) |
| base-b-test-4-npu-a3 / run (1) | 50.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545428) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 50.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545742) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 50.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545878) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 50.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545890) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 50.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545892) |

- **multimodal-gen-test-1-npu-a3**: 日志中未出现测试失败或错误信息，仅显示上传diffusion-failures目录时无文件，可能测试通过或未生成失败产物，作业失败原因需进一步查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545067

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545256

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象缺失，可能是构建产物未上传、路径配置错误或存储被清理，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545272

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545295

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/缓存文件在存储中缺失，可能是由于文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545346

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存、模型权重或日志文件缺失，需检查相关资源是否存在或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545424

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545428

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545742

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545878

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在 Azure Blob 存储中找到，可能是资源被清理或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545890

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545892

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32308517891/job/96246545319) |


## [Run #32308365479](https://github.com/sgl-project/sglang/actions/runs/32308365479)
- **分支**: `junshen/fix-nul-grammar-segfault`
- **总耗时**: 14.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32308365479

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 环境问题 | Git 拉取 PR 合并提交失败，远端仓库不存在该 ref。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246075998) |
| base-b-test-4-npu-a3 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076109) |
| base-b-test-4-npu-a3 / run (1) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076117) |
| base-b-test-2-npu-a3 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076125) |
| base-b-test-16-npu-a3 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076177) |
| base-b-test-8-npu-a3 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076324) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076457) |
| base-b-test-1-npu-a3 / run (0) | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076542) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076555) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076579) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 10.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076691) |

- **multimodal-gen-test-1-npu-a3**: checkout 阶段执行 git fetch 时，远端返回 "not our ref 3c8d0a8e..."，重试三次均失败，导致作业无法获取代码而终止。可能是 PR 已关闭或 ref 过期，属于基础设施/仓库状态问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246075998

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是资源被清理或路径错误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076109

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076117

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076125

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载日志文件，但 Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施或配置问题，与代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076177

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076324

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储中找到，可能是资源被清理、路径错误或上传失败，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076457

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076542

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076555

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076579

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076691

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32308365479/job/96246076266) |


## [Run #32306313136](https://github.com/sgl-project/sglang/actions/runs/32306313136)
- **分支**: `rainj-me/rust-server-refactor2`
- **总耗时**: 19.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32306313136

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777619) |
| base-b-test-16-npu-a3 / run (0) | 18.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777737) |
| base-b-test-8-npu-a3 / run (0) | 18.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777739) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777740) |
| base-b-test-1-npu-a3 / run (0) | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777773) |
| base-b-test-4-npu-a3 / run (1) | 18.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777800) |
| base-b-test-4-npu-a3 / run (0) | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777888) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241778081) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241778194) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241778233) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 18.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241778276) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777619

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该文件不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误导致，属于基础设施或环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777737

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误，或资源被清理。属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777739

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777740

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777773

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777800

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777888

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241778081

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241778194

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或被误删，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241778233

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241778276

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241756378) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32306313136/job/96241777736) |


## [Run #32305460556](https://github.com/sgl-project/sglang/actions/runs/32305460556)
- **分支**: `feat/parallel-stage-groups`
- **总耗时**: 67.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32305460556

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32305460556/job/96237236430) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时未找到文件，无法判断测试失败的具体原因，可能是测试未运行或日志被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32305460556/job/96237236430


## [Run #32305203302](https://github.com/sgl-project/sglang/actions/runs/32305203302)
- **分支**: `codex/minimax-h3-reference-rng`
- **总耗时**: 71.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32305203302

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32305203302/job/96236463555) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32305203302/job/96236463555


## [Run #32304831260](https://github.com/sgl-project/sglang/actions/runs/32304831260)
- **分支**: `junshen/fix-nul-grammar-segfault`
- **总耗时**: 43.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32304831260

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.0min | 环境问题 | Git 拉取失败，远端仓库缺少指定 commit 引用 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319282) |
| base-b-test-8-npu-a3 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319471) |
| base-b-test-1-npu-a3 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319518) |
| base-b-test-4-npu-a3 / run (1) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319620) |
| base-b-test-4-npu-a3 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319630) |
| base-b-test-2-npu-a3 / run (0) | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319649) |
| base-b-test-16-npu-a3 / run (0) | 42.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319657) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235320274) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235320313) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235320485) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 42.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235320583) |

- **multimodal-gen-test-1-npu-a3**: checkout 时 fetch 指定 commit 4994bee 失败，远端返回 'not our ref'，重试三次均失败，导致作业无法开始，属于仓库或缓存同步问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319282

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319471

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319518

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319620

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319630

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319649

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319657

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被清理，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235320274

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235320313

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或配置变更所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235320485

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235320583

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32304831260/job/96235319464) |


## [Run #32304721645](https://github.com/sgl-project/sglang/actions/runs/32304721645)
- **分支**: `sglang_pp_bug4`
- **总耗时**: 131.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32304721645

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志被截断，未显示实际测试结果，仅见上传artifact时无失败文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008749) |
| base-b-test-1-npu-a3 / run (0) | 130.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008795) |
| base-b-test-2-npu-a3 / run (0) | 130.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008844) |
| base-b-test-8-npu-a3 / run (0) | 130.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008897) |
| base-b-test-16-npu-a3 / run (0) | 130.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008929) |
| base-b-test-4-npu-a3 / run (1) | 130.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009045) |
| base-b-test-4-npu-a3 / run (0) | 130.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009088) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 130.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009378) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 130.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009390) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 130.9min | 环境问题 | CI作业因Azure Blob存储中找不到指定文件而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009410) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 130.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009469) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法判断测试是否通过或失败。仅看到上传diffusion-failures目录时提示无文件，说明可能测试通过或失败原因未记录。需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008749

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008795

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008844

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008897

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象缺失，可能是日志上传或依赖下载路径错误，属于基础设施配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008929

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是缓存或上传步骤失败，需检查相关存储路径及上传逻辑。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009045

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009088

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009378

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009390

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示BlobNotFound错误，说明作业依赖的某个文件（如模型权重或测试数据）在存储中不存在，可能是上传缺失或路径配置错误，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009410

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境配置或资源缺失问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235009469

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32304721645/job/96235008803) |


## [Run #32304639300](https://github.com/sgl-project/sglang/actions/runs/32304639300)
- **分支**: `codex/minimax-h3-ref2va-audio-ci`
- **总耗时**: 66.8min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32304639300

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32304639300/job/96234724868) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。末尾显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32304639300/job/96234724868


## [Run #32302940992](https://github.com/sgl-project/sglang/actions/runs/32302940992)
- **分支**: `lsyin/page-tail-write`
- **总耗时**: 28.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32302940992

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 15.4min | 环境问题 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230892881) |
| base-b-test-4-npu-a3 / run (1) | 17.9min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893072) |
| base-b-test-8-npu-a3 / run (0) | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893075) |
| base-b-test-2-npu-a3 / run (0) | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893137) |
| base-b-test-4-npu-a3 / run (0) | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893162) |
| base-b-test-1-npu-a3 / run (0) | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893175) |
| base-b-test-16-npu-a3 / run (0) | 17.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893300) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893395) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893534) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893607) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893621) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230892881

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893072

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893075

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893137

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893162

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是缓存或依赖文件未正确上传，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893175

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893300

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893395

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893534

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893607

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893621

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32302940992/job/96230893280) |


## [Run #32302865983](https://github.com/sgl-project/sglang/actions/runs/32302865983)
- **分支**: `main`
- **总耗时**: 45.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32302865983

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 37.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384384) |
| base-b-test-1-npu-a3 / run (0) | 40.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384586) |
| base-b-test-8-npu-a3 / run (0) | 40.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384635) |
| base-b-test-4-npu-a3 / run (1) | 40.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384712) |
| base-b-test-2-npu-a3 / run (0) | 40.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384722) |
| base-b-test-4-npu-a3 / run (0) | 40.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384790) |
| base-b-test-16-npu-a3 / run (0) | 40.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384801) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230385148) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 40.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230385186) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 40.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230385277) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 40.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230385293) |

- **multimodal-gen-test-1-npu-a3**: 日志截取部分仅包含GitHub Actions初始化、Node版本警告及上传diffusion-failures工件（未找到文件），未包含multimodal-gen测试的实际执行和失败信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384384

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是上传失败或路径配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384586

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384635

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384712

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384722

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384790

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384801

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230385148

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230385186

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230385277

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230385293

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32302865983/job/96230384706) |


## [Run #32302615903](https://github.com/sgl-project/sglang/actions/runs/32302615903)
- **分支**: `kpham/mamba-track-interval-tree-page`
- **总耗时**: 23.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32302615903

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.5min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和上传步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228496896) |
| base-a-test-1-npu-a2 / run (0) | 4.7min | 代码错误 | 测试套件注册了无效的suite名称，导致校验失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497278) |
| base-b-test-16-npu-a3 / run (0) | 17.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497346) |
| base-b-test-4-npu-a3 / run (1) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497406) |
| base-b-test-2-npu-a3 / run (0) | 17.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497453) |
| base-b-test-4-npu-a3 / run (0) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497454) |
| base-b-test-1-npu-a3 / run (0) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497493) |
| base-b-test-8-npu-a3 / run (0) | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497564) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228498100) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228498141) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228498175) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 17.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228498275) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含实际测试执行或失败信息，仅显示runner启动、依赖下载和上传artifacts（无文件）。可能测试未运行或日志被截断，需查看完整日志确认失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228496896

- **base-a-test-1-npu-a2 / run (0)**: test/registered/ep/test_routed_experts_dp_readback.py 中注册的 suite 为 'base-c-test-deepep-8-gpu-h200'，但该 suite 不在当前作业的测试套件列表中，触发 ValueError，导致 CI 失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497278

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497346

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497406

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497453

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497454

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497493

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/环境配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228497564

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228498100

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228498141

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228498175

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32302615903/job/96228498275


## [Run #32301099404](https://github.com/sgl-project/sglang/actions/runs/32301099404)
- **分支**: `main`
- **总耗时**: 24.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32301099404

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223833826) |
| base-b-test-1-npu-a3 / run (0) | 18.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223833946) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | 代码错误 | 测试注册到无效的测试套件导致校验失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834003) |
| base-b-test-2-npu-a3 / run (0) | 18.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834019) |
| base-b-test-4-npu-a3 / run (0) | 18.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834020) |
| base-b-test-16-npu-a3 / run (0) | 18.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834051) |
| base-b-test-8-npu-a3 / run (0) | 18.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834056) |
| base-b-test-4-npu-a3 / run (1) | 18.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834270) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 18.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834393) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 18.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834461) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 18.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834539) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 18.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834581) |

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（无文件上传），未出现任何测试执行或失败断言信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223833826

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223833946

- **base-a-test-1-npu-a2 / run (0)**: run_suite.py在validate_all_suites阶段发现test/routed_experts_dp_readback.py注册的backend=CUDA与套件'base-c-test-deepep-8-gpu-h200'不匹配，抛出ValueError，导致作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834003

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834019

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源（如模型权重、缓存或日志）已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834020

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834051

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834056

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834270

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834393

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834461

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834539

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223834581

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32301099404/job/96223610933) |


## [Run #32299572780](https://github.com/sgl-project/sglang/actions/runs/32299572780)
- **分支**: `main`
- **总耗时**: 17.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32299572780

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220372902) |
| base-b-test-4-npu-a3 / run (0) | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220372968) |
| base-b-test-2-npu-a3 / run (0) | 11.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373046) |
| base-b-test-4-npu-a3 / run (1) | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373073) |
| base-a-test-1-npu-a2 / run (0) | 4.5min | 代码错误 | 测试套件校验失败，测试文件注册到了无效的套件 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373078) |
| base-b-test-16-npu-a3 / run (0) | 11.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373079) |
| base-b-test-1-npu-a3 / run (0) | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373136) |
| base-b-test-8-npu-a3 / run (0) | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373143) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373525) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373663) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373665) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373771) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220372902

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220372968

- **base-b-test-2-npu-a3 / run (0)**: 作业失败原因是BlobNotFound错误，即请求的blob在存储中不存在，可能是资源被删除、路径错误或上传未完成，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373046

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373073

- **base-a-test-1-npu-a2 / run (0)**: test/registered/ep/test_routed_experts_dp_readback.py 的 backend=CUDA，但被注册到 'base-c-test-deepep-8-gpu-h200' 套件，该套件可能不适用于当前 NPU 环境，导致 run_suite.py 抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373078

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象已被删除或路径错误，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373079

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是缓存或依赖文件未正确上传，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373136

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373143

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373525

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373663

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373665

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32299572780/job/96220373771


## [Run #32298749607](https://github.com/sgl-project/sglang/actions/runs/32298749607)
- **分支**: `lsyin/page-tail-write`
- **总耗时**: 51.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32298749607

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 38.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757736) |
| base-b-test-8-npu-a3 / run (0) | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757774) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | 代码错误 | 测试文件注册到无效的测试套件导致校验失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757796) |
| base-b-test-4-npu-a3 / run (0) | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757824) |
| base-b-test-1-npu-a3 / run (0) | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757849) |
| base-b-test-2-npu-a3 / run (0) | 44.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757850) |
| base-b-test-16-npu-a3 / run (0) | 44.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757891) |
| base-b-test-4-npu-a3 / run (1) | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757926) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216758145) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216758155) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216758180) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 44.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216758215) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757736

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757774

- **base-a-test-1-npu-a2 / run (0)**: run_suite.py 在 validate_all_suites 阶段发现 test/registered/ep/test_routed_experts_dp_readback.py 被注册到 backend=CUDA 的 'base-c-test-deepep-8-gpu-h200' 套件，但当前作业是 NPU 环境，套件不匹配，触发 ValueError 异常，导致 CI 失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757796

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的工件/缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757824

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757849

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757850

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757891

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216757926

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216758145

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216758155

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216758180

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被删除或配置不一致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216758215

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 1.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32298749607/job/96216391203) |


## [Run #32298274193](https://github.com/sgl-project/sglang/actions/runs/32298274193)
- **分支**: `lsyin/page-tail-write`
- **总耗时**: 6.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32298274193

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781356) |
| base-b-test-4-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781631) |
| base-b-test-4-npu-a3 / run (1) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781644) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | 代码错误 | 测试套件注册了无效的测试文件，导致校验失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781708) |
| base-b-test-16-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781743) |
| base-b-test-2-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781805) |
| base-b-test-1-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781809) |
| base-b-test-8-npu-a3 / run (0) | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781868) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214782164) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214782175) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214782238) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214782315) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781356

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781631

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781644

- **base-a-test-1-npu-a2 / run (0)**: test/registered/ep/test_routed_experts_dp_readback.py 被注册到 backend=CUDA 的 'base-c-test-deepep-8-gpu-h200' 套件，但该套件不属于当前 NPU 作业，导致 validate_all_suites 抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781708

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781743

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781805

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781809

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214781868

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214782164

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214782175

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214782238

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32298274193/job/96214782315


## [Run #32296985048](https://github.com/sgl-project/sglang/actions/runs/32296985048)
- **分支**: `main`
- **总耗时**: 33.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32296985048

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.6min | 其他 | 作业未显示明确失败原因，日志仅包含环境警告和上传空产物提示。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210637954) |
| base-b-test-16-npu-a3 / run (0) | 27.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638098) |
| base-a-test-1-npu-a2 / run (0) | 4.8min | 代码错误 | 测试注册到无效套件导致校验失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638107) |
| base-b-test-8-npu-a3 / run (0) | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638151) |
| base-b-test-4-npu-a3 / run (1) | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638172) |
| base-b-test-1-npu-a3 / run (0) | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638242) |
| base-b-test-4-npu-a3 / run (0) | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638265) |
| base-b-test-2-npu-a3 / run (0) | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638291) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638453) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638475) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638489) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 27.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638556) |

- **multimodal-gen-test-1-npu-a3**: 日志中未见测试失败或错误信息，仅有Node 20弃用警告和diffusion-failures目录无文件上传提示，可能作业被取消或日志截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210637954

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/日志文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638098

- **base-a-test-1-npu-a2 / run (0)**: run_suite.py 校验发现 test/registered/ep/test_routed_experts_dp_readback.py 被注册到 backend=CUDA 的 'base-c-test-deepep-8-gpu-h200' 套件，但当前作业为 NPU 环境，套件不匹配，触发 ValueError 异常。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638107

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638151

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638172

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638242

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638265

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源（如模型权重、缓存或日志）已被删除或路径错误，属于环境或资源配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638291

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638453

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638475

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638489

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、数据集或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296985048/job/96210638556


## [Run #32296305056](https://github.com/sgl-project/sglang/actions/runs/32296305056)
- **分支**: `main`
- **总耗时**: 7.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32296305056

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491648) |
| base-b-test-16-npu-a3 / run (0) | 6.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491799) |
| base-b-test-4-npu-a3 / run (0) | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491837) |
| base-a-test-1-npu-a2 / run (0) | 4.7min | 代码错误 | 测试套件注册了无效的测试文件，导致校验失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491843) |
| base-b-test-4-npu-a3 / run (1) | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491882) |
| base-b-test-1-npu-a3 / run (0) | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491914) |
| base-b-test-2-npu-a3 / run (0) | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491938) |
| base-b-test-8-npu-a3 / run (0) | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492009) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492122) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492166) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492256) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492303) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491648

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491799

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491837

- **base-a-test-1-npu-a2 / run (0)**: test/registered/ep/test_routed_experts_dp_readback.py 被注册到 backend=CUDA 的 suite 'base-c-test-deepep-8-gpu-h200'，但当前作业是 NPU 环境，套件与后端不匹配，run_suite.py 校验时抛出 ValueError。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491843

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491882

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491914

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208491938

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492009

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492122

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492166

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492256

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296305056/job/96208492303


## [Run #32296057568](https://github.com/sgl-project/sglang/actions/runs/32296057568)
- **分支**: `jialino/rust-tree-core-full`
- **总耗时**: 58.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32296057568

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 56.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613291) |
| base-b-test-16-npu-a3 / run (0) | 57.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613495) |
| base-b-test-1-npu-a3 / run (0) | 57.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613533) |
| base-a-test-1-npu-a2 / run (0) | 5.3min | 代码错误 | 测试文件缺少main入口导致测试收集失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613568) |
| base-b-test-4-npu-a3 / run (0) | 57.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613648) |
| base-b-test-8-npu-a3 / run (0) | 57.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613653) |
| base-b-test-4-npu-a3 / run (1) | 57.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613657) |
| base-b-test-2-npu-a3 / run (0) | 57.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613695) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 57.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207614095) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 57.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207614158) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 57.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207614205) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 57.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207614242) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本或日志被截断，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613291

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，与代码或测试本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613495

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613533

- **base-a-test-1-npu-a2 / run (0)**: test/registered/unit/mem_cache/test_rust_tree_core.py缺少`if __name__ == "__main__":`入口，导致pytest风格测试在`python3 file.py -f`方式下静默跳过，collect_tests抛出ValueError，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613568

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613648

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613653

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613657

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207613695

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 存储中缺失或路径错误，可能是上传失败或配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207614095

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207614158

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207614205

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32296057568/job/96207614242


## [Run #32294556347](https://github.com/sgl-project/sglang/actions/runs/32294556347)
- **分支**: `codex/minimax-h3-cube-sparse-attn`
- **总耗时**: 79.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32294556347

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294556347/job/96202862080) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294556347/job/96202862080


## [Run #32294456267](https://github.com/sgl-project/sglang/actions/runs/32294456267)
- **分支**: `main`
- **总耗时**: 20.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32294456267

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 18.5min | 环境问题 | 作业因环境问题失败，未生成失败产物。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631312) |
| base-b-test-4-npu-a3 / run (0) | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631351) |
| base-b-test-4-npu-a3 / run (1) | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631352) |
| base-b-test-2-npu-a3 / run (0) | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631370) |
| base-b-test-1-npu-a3 / run (0) | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631405) |
| base-a-test-1-npu-a2 / run (0) | 4.6min | 代码错误 | 测试注册到无效的suite，导致测试套件校验失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631473) |
| base-b-test-16-npu-a3 / run (0) | 19.4min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631474) |
| base-b-test-8-npu-a3 / run (0) | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631506) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631817) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631919) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631929) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202632000) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在运行后未找到diffusion-failures目录，上传产物时提示无文件，可能因NPU环境或测试配置问题导致测试未执行或失败未记录。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631312

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631351

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631352

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631370

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631405

- **base-a-test-1-npu-a2 / run (0)**: run_suite.py在validate_all_suites阶段发现test/routed_experts_dp_readback.py被注册到backend=CUDA的suite 'base-c-test-deepep-8-gpu-h200'，但当前作业是NPU环境，suite不匹配，抛出ValueError导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631473

- **base-b-test-16-npu-a3 / run (0)**: 作业日志显示BlobNotFound错误，说明CI流程尝试下载的某个文件（如模型权重、测试数据或缓存）在Azure Blob存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631474

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或构建产物）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被清理或配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631506

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631817

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631919

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202631929

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294456267/job/96202632000


## [Run #32294029282](https://github.com/sgl-project/sglang/actions/runs/32294029282)
- **分支**: `codex/minimax-h3-ref2va-audio-ci`
- **总耗时**: 12.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32294029282

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.9min | 环境问题 | GitHub Actions 拉取 PR 合并提交时，远端仓库缺少该 ref，导致 checkout 失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32294029282/job/96201208281) |

- **multimodal-gen-test-1-npu-a3**: 作业在 fetch PR 合并提交 1be7945 时，git 远端返回 'not our ref'，重试三次均失败，属于临时性远端仓库状态或缓存不一致问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32294029282/job/96201208281


## [Run #32291208807](https://github.com/sgl-project/sglang/actions/runs/32291208807)
- **分支**: `fix/single-rank-nccl-vram`
- **总耗时**: 67.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32291208807

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志被截断，未显示实际测试结果，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32291208807/job/96192297854) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法判断测试是否失败。仅能看到上传diffusion-failures目录时提示无文件，说明可能没有失败用例，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32291208807/job/96192297854


## [Run #32290707158](https://github.com/sgl-project/sglang/actions/runs/32290707158)
- **分支**: `main`
- **总耗时**: 40.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32290707158

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846040) |
| base-b-test-1-npu-a3 / run (0) | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846373) |
| base-b-test-2-npu-a3 / run (0) | 39.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846394) |
| base-b-test-8-npu-a3 / run (0) | 39.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846410) |
| base-a-test-1-npu-a2 / run (0) | 3.9min | 环境问题 | rustup 下载中断导致环境准备失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846497) |
| base-b-test-4-npu-a3 / run (0) | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846549) |
| base-b-test-4-npu-a3 / run (1) | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846561) |
| base-b-test-16-npu-a3 / run (0) | 39.5min | 环境问题 | 日志中引用的Azure Blob存储资源不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846649) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846771) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846783) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846965) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 39.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846980) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846040

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846373

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846394

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或资源被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846410

- **base-a-test-1-npu-a2 / run (0)**: 在安装 Rust 时，从内部缓存服务器下载 rustup-init 过程中连接中断（curl 错误 18），剩余 1436225 字节未读取，导致脚本退出码 18，作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846497

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846549

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846561

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的blob（可能为测试数据或模型权重）已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846649

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846771

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846783

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846965

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190846980

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32290707158/job/96190621852) |


## [Run #32289853679](https://github.com/sgl-project/sglang/actions/runs/32289853679)
- **分支**: `main`
- **总耗时**: 8.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32289853679

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122233) |
| base-b-test-2-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122465) |
| base-b-test-16-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122475) |
| base-a-test-1-npu-a2 / run (0) | 4.9min | 代码错误 | 测试注册到无效套件导致校验失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122491) |
| base-b-test-8-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122524) |
| base-b-test-4-npu-a3 / run (1) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122552) |
| base-b-test-4-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122639) |
| base-b-test-1-npu-a3 / run (0) | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122645) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122836) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122945) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188123054) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188123346) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程资源（如模型权重或数据文件）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122233

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122465

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是因为日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122475

- **base-a-test-1-npu-a2 / run (0)**: run_suite.py 的 validate_all_suites 发现 test/registered/ep/test_routed_experts_dp_readback.py 的 backend=CUDA 被注册到 'base-c-test-deepep-8-gpu-h200' 套件，但该套件不在当前 NPU 作业的套件列表中，触发 ValueError，导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122491

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122524

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122552

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是构建产物未上传或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122639

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122645

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122836

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188122945

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188123054

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32289853679/job/96188123346


## [Run #32287709453](https://github.com/sgl-project/sglang/actions/runs/32287709453)
- **分支**: `cctry/hicache/per-host-memory-budget`
- **总耗时**: 42.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32287709453

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 34.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559516) |
| base-b-test-1-npu-a3 / run (0) | 39.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559632) |
| base-b-test-8-npu-a3 / run (0) | 39.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559677) |
| base-b-test-16-npu-a3 / run (0) | 39.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559717) |
| base-b-test-4-npu-a3 / run (0) | 39.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559718) |
| base-b-test-4-npu-a3 / run (1) | 39.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559819) |
| base-b-test-2-npu-a3 / run (0) | 39.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559881) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181560104) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 39.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181560248) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 39.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181560319) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 39.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181560394) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559516

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559632

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559677

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，可能是构建产物或依赖缓存缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559717

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559718

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559819

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559881

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、缓存或日志）在 Azure Blob 中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181560104

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181560248

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181560319

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181560394

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181312898) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32287709453/job/96181559786) |


## [Run #32287282126](https://github.com/sgl-project/sglang/actions/runs/32287282126)
- **分支**: `cctry/hicache/per-host-memory-budget`
- **总耗时**: 5.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32287282126

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821042) |
| base-a-test-1-npu-a2 / run (0) | 4.4min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821046) |
| base-b-test-4-npu-a3 / run (1) | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821082) |
| base-b-test-2-npu-a3 / run (0) | 4.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821123) |
| base-b-test-16-npu-a3 / run (0) | 4.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821166) |
| base-b-test-8-npu-a3 / run (0) | 4.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821176) |
| base-b-test-4-npu-a3 / run (0) | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821233) |
| base-b-test-1-npu-a3 / run (0) | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821273) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821580) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821624) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821684) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 4.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821707) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821042

- **base-a-test-1-npu-a2 / run (0)**: 日志显示执行自定义容器实现时失败，提示联系自托管runner管理员，属于NPU CI基础设施环境问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821046

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821082

- **base-b-test-2-npu-a3 / run (0)**: 作业尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821123

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的存储对象已被删除或路径错误，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821166

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821176

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821233

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821273

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是构建产物未上传或存储配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821580

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821624

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821684

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被清理，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32287282126/job/96179821707


## [Run #32286037968](https://github.com/sgl-project/sglang/actions/runs/32286037968)
- **分支**: `main`
- **总耗时**: 40.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32286037968

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 22.8min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839618) |
| base-b-test-4-npu-a3 / run (1) | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839708) |
| base-b-test-2-npu-a3 / run (0) | 39.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839746) |
| base-b-test-4-npu-a3 / run (0) | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839828) |
| base-b-test-16-npu-a3 / run (0) | 39.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839947) |
| base-b-test-1-npu-a3 / run (0) | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839954) |
| base-b-test-8-npu-a3 / run (0) | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840033) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840852) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840962) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840971) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 39.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840992) |

- **multimodal-gen-test-1-npu-a3**: 作业在运行测试后上传diffusion-failures目录时提示无文件，但日志中间部分被省略，无法判断是测试未执行、全部通过还是失败原因未记录。需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839618

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839708

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或文件被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839746

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839828

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储账户中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839947

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839954

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840033

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840852

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840962

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840971

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175840992

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32286037968/job/96175839814) |


## [Run #32284657305](https://github.com/sgl-project/sglang/actions/runs/32284657305)
- **分支**: `main`
- **总耗时**: 14.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32284657305

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447730) |
| base-b-test-16-npu-a3 / run (0) | 13.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447748) |
| base-b-test-2-npu-a3 / run (0) | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447774) |
| base-b-test-8-npu-a3 / run (0) | 13.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447845) |
| base-b-test-1-npu-a3 / run (0) | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447918) |
| base-b-test-4-npu-a3 / run (0) | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447920) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447927) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171448294) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171448312) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171448378) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 13.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171448449) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的模型或数据文件在 Azure Blob 存储中缺失，可能是文件被误删或路径配置错误，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447730

- **base-b-test-16-npu-a3 / run (0)**: 作业在下载或访问Azure Blob存储中的某个blob时失败，返回BlobNotFound错误。这通常是因为文件被删除、路径错误或存储配置变更，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447748

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447774

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志或依赖文件未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447845

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447918

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查相关 blob 是否存在及路径配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447920

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447927

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171448294

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171448312

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171448378

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或数据在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171448449

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32284657305/job/96171447891) |


## [Run #32282822097](https://github.com/sgl-project/sglang/actions/runs/32282822097)
- **分支**: `codex/minimax-h3-ref2va-audio-ci`
- **总耗时**: 68.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32282822097

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 64.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32282822097/job/96165433885) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出，仅显示上传diffusion-failures目录时无文件，可能测试未运行或未产生失败文件，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32282822097/job/96165433885


## [Run #32280512153](https://github.com/sgl-project/sglang/actions/runs/32280512153)
- **分支**: `agentx-upstream/eagle-selected-row-logits-20260819`
- **总耗时**: 43.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32280512153

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 30.3min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171411917) |
| base-b-test-4-npu-a3 / run (0) | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412240) |
| base-b-test-16-npu-a3 / run (0) | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412333) |
| base-b-test-4-npu-a3 / run (1) | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412337) |
| base-b-test-8-npu-a3 / run (0) | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412358) |
| base-b-test-1-npu-a3 / run (0) | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412419) |
| base-b-test-2-npu-a3 / run (0) | 43.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412425) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412729) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412786) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412902) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 43.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171413040) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或失败原因被截断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171411917

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412240

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或资源配置问题，需检查存储路径或重新上传产物。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412333

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412337

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412358

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412419

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412425

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412729

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412786

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412902

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171413040

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32280512153/job/96171412341) |


## [Run #32274443191](https://github.com/sgl-project/sglang/actions/runs/32274443191)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 102.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32274443191

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.1min | 其他 | 作业日志不完整，未显示实际测试结果，仅包含环境准备和上传失败信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770504) |
| base-b-test-2-npu-a3 / run (0) | 100.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770597) |
| base-b-test-4-npu-a3 / run (0) | 100.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770600) |
| base-b-test-4-npu-a3 / run (1) | 100.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770609) |
| base-b-test-1-npu-a3 / run (0) | 100.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770718) |
| base-b-test-8-npu-a3 / run (0) | 100.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770802) |
| base-b-test-16-npu-a3 / run (0) | 100.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770887) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 100.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138771174) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 100.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138771215) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 100.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138771216) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 100.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138771231) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行过程或失败断言，仅显示上传diffusion-failures目录时无文件，可能测试未运行或日志被截断，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770504

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770597

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是构建产物未上传或路径错误，属于环境配置或存储问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770600

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770609

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770718

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770802

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770887

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138771174

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138771215

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，属于环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138771216

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138771231

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32274443191/job/96138770708) |


## [Run #32274079488](https://github.com/sgl-project/sglang/actions/runs/32274079488)
- **分支**: `fix/single-rank-nccl-vram`
- **总耗时**: 75.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32274079488

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32274079488/job/96166082456) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或失败原因未记录。
  链接: https://github.com/sgl-project/sglang/actions/runs/32274079488/job/96166082456


## [Run #32273609126](https://github.com/sgl-project/sglang/actions/runs/32273609126)
- **分支**: `xpu-kernel-cache-image-id`
- **总耗时**: 235.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32273609126

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.0min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135837261) |
| base-b-test-2-npu-a3 / run (0) | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838239) |
| base-b-test-8-npu-a3 / run (0) | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838255) |
| base-b-test-16-npu-a3 / run (0) | 234.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838385) |
| base-b-test-4-npu-a3 / run (0) | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838427) |
| base-b-test-4-npu-a3 / run (1) | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838544) |
| base-b-test-1-npu-a3 / run (0) | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838589) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135840296) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135840349) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135840446) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 234.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135840450) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135837261

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838239

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838255

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838385

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838427

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838544

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是构建产物或缓存未正确上传，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838589

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135840296

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135840349

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135840446

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135840450

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32273609126/job/96135838267) |


## [Run #32273468158](https://github.com/sgl-project/sglang/actions/runs/32273468158)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 7.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32273468158

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385243) |
| base-b-test-8-npu-a3 / run (0) | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385320) |
| base-b-test-4-npu-a3 / run (1) | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385381) |
| base-b-test-4-npu-a3 / run (0) | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385399) |
| base-b-test-1-npu-a3 / run (0) | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385502) |
| base-a-test-1-npu-a2 / run (0) | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385570) |
| base-b-test-16-npu-a3 / run (0) | 6.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385599) |
| base-b-test-2-npu-a3 / run (0) | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385654) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135386507) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135386596) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135386676) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135386858) |

- **multimodal-gen-test-1-npu-a3**: 错误码BlobNotFound表明CI作业依赖的某个文件（如模型权重或测试数据）在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385243

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385320

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385381

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385399

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385502

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385570

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385599

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135385654

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135386507

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135386596

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径配置错误，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135386676

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32273468158/job/96135386858


## [Run #32272583378](https://github.com/sgl-project/sglang/actions/runs/32272583378)
- **分支**: `fix/dsa-movekv-page-aware`
- **总耗时**: 46.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32272583378

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 24.5min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459343) |
| base-b-test-4-npu-a3 / run (1) | 40.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459363) |
| base-b-test-16-npu-a3 / run (0) | 40.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459372) |
| base-b-test-8-npu-a3 / run (0) | 40.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459456) |
| base-b-test-2-npu-a3 / run (0) | 40.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459517) |
| base-b-test-1-npu-a3 / run (0) | 40.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459528) |
| base-b-test-4-npu-a3 / run (0) | 40.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459563) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 40.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132460030) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 40.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132460031) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 40.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132460034) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132460062) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤。无法判断测试是否失败或失败原因，可能为日志截断或作业被提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459343

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459363

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459372

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459456

- **base-b-test-2-npu-a3 / run (0)**: 作业在下载或访问某个blob时返回BlobNotFound错误，可能是CI配置中引用的文件被删除、路径错误或存储账户配置变更，属于环境或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459517

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459528

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459563

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132460030

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132460031

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132460034

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132460062

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 10.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32272583378/job/96132459783) |


## [Run #32272089424](https://github.com/sgl-project/sglang/actions/runs/32272089424)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 12.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32272089424

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130883787) |
| base-b-test-4-npu-a3 / run (0) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884832) |
| base-b-test-2-npu-a3 / run (0) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884853) |
| base-b-test-8-npu-a3 / run (0) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884894) |
| base-b-test-16-npu-a3 / run (0) | 11.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884995) |
| base-b-test-4-npu-a3 / run (1) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130885055) |
| base-b-test-1-npu-a3 / run (0) | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130885149) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130886071) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130886238) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130886506) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130886558) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130883787

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884832

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884853

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884894

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、文件被清理或配置问题，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884995

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源缺失，可能是上传失败、路径错误或资源被清理，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130885055

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的工件/缓存文件已被删除或路径错误，属于外部存储依赖问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130885149

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130886071

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130886238

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130886506

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130886558

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32272089424/job/96130884881) |


## [Run #32271980800](https://github.com/sgl-project/sglang/actions/runs/32271980800)
- **分支**: `nemotron-3.5-spec-comparison`
- **总耗时**: 204.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32271980800

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 62.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132126793) |
| base-b-test-2-npu-a3 / run (0) | 198.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127059) |
| base-a-test-1-npu-a2 / run (0) | 10.9min | 环境问题 | NPU测试用例执行失败，返回未知异常错误码。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127112) |
| base-b-test-16-npu-a3 / run (0) | 198.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127142) |
| base-b-test-4-npu-a3 / run (0) | 198.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127167) |
| base-b-test-4-npu-a3 / run (1) | 198.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127171) |
| base-b-test-8-npu-a3 / run (0) | 198.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127216) |
| base-b-test-1-npu-a3 / run (0) | 198.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127271) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 198.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127804) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 198.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127864) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 198.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132128002) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 198.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132128124) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132126793

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于环境或配置问题，需检查资源路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127059

- **base-a-test-1-npu-a2 / run (0)**: 测试文件test_npu_ascend_backend.py运行23秒后失败，日志显示ERR99999 UNKNOWN application exception，可能是NPU环境或依赖问题导致，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127112

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127142

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127167

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127171

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127216

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是资源被清理、路径错误或上传失败，属于环境配置或依赖资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127271

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127804

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132127864

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132128002

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271980800/job/96132128124


## [Run #32271002353](https://github.com/sgl-project/sglang/actions/runs/32271002353)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 11.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32271002353

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350265) |
| base-b-test-2-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350441) |
| base-b-test-1-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350503) |
| base-b-test-4-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350539) |
| base-b-test-4-npu-a3 / run (1) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350558) |
| base-b-test-8-npu-a3 / run (0) | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350565) |
| base-b-test-16-npu-a3 / run (0) | 10.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350570) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350971) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127351108) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127351147) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 10.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127351205) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350265

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传失败、路径错误或存储被清理所致，属于基础设施或环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350441

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350503

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境配置或资源缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350539

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350558

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350565

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失或配置错误，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350570

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350971

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127351108

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被清理、路径错误或上传失败，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127351147

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127351205

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32271002353/job/96127350692) |


## [Run #32268787864](https://github.com/sgl-project/sglang/actions/runs/32268787864)
- **分支**: `main`
- **总耗时**: 164.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32268787864

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659781) |
| base-b-test-16-npu-a3 / run (0) | 158.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659787) |
| base-b-test-1-npu-a3 / run (0) | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659907) |
| base-b-test-4-npu-a3 / run (1) | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659978) |
| base-b-test-8-npu-a3 / run (0) | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660127) |
| multimodal-gen-test-1-npu-a3 | 63.5min | 其他 | 作业日志不完整，未显示测试失败的具体原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660146) |
| base-b-test-2-npu-a3 / run (0) | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660379) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660985) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660993) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121661150) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 158.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121661426) |

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理或路径错误，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659781

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659787

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659907

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659978

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更导致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660127

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本警告及上传失败产物（无文件）等常规信息，未出现测试执行、断言失败或错误堆栈，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660146

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660379

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是资源清理或路径配置错误，需检查相关存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660985

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查相关 blob 的可用性。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121660993

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121661150

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121661426

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32268787864/job/96121659673) |


## [Run #32266887725](https://github.com/sgl-project/sglang/actions/runs/32266887725)
- **分支**: `minimax-h3-on-npu-support`
- **总耗时**: 40.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32266887725

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613577) |
| base-b-test-2-npu-a3 / run (0) | 35.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613846) |
| base-b-test-4-npu-a3 / run (1) | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613924) |
| base-b-test-8-npu-a3 / run (0) | 35.0min | 环境问题 | CI作业因Azure Blob存储中指定的blob不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613937) |
| base-b-test-16-npu-a3 / run (0) | 35.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614084) |
| base-b-test-1-npu-a3 / run (0) | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614153) |
| base-b-test-4-npu-a3 / run (0) | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614463) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614516) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614574) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614677) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 35.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614681) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node.js弃用警告及上传diffusion-failures工件时未找到文件，未包含multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613577

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是日志保留策略或上传失败所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613846

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613924

- **base-b-test-8-npu-a3 / run (0)**: 日志显示BlobNotFound错误，表明作业尝试下载或访问的Azure Blob存储资源已被删除或路径错误，属于基础设施/环境配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613937

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614084

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614153

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614463

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614516

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，请求的资源在 Azure Blob 存储中未找到，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614574

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614677

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113614681

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32266887725/job/96113613736) |


## [Run #32266295648](https://github.com/sgl-project/sglang/actions/runs/32266295648)
- **分支**: `main`
- **总耗时**: 29.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32266295648

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111662889) |
| multimodal-gen-test-1-npu-a3 | 14.2min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111662913) |
| base-b-test-4-npu-a3 / run (1) | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663011) |
| base-b-test-2-npu-a3 / run (0) | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663048) |
| base-b-test-4-npu-a3 / run (0) | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663232) |
| base-b-test-8-npu-a3 / run (0) | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663284) |
| base-b-test-16-npu-a3 / run (0) | 24.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663305) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663552) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663573) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663720) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 24.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663912) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、数据集或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传、被删除或配置的 URL 有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111662889

- **multimodal-gen-test-1-npu-a3**: 日志中仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤（无文件上传），未出现测试执行或失败的具体错误信息，无法判断失败根因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111662913

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663011

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于外部存储环境问题，需检查相关 blob 是否存在或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663048

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物或依赖文件未正确上传或已被删除，属于环境或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663232

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663284

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663305

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或缓存）在 Azure Blob 中缺失或路径错误，可能是资源被清理、上传失败或配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663552

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663573

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663720

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111663912

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32266295648/job/96111662926) |


## [Run #32266199885](https://github.com/sgl-project/sglang/actions/runs/32266199885)
- **分支**: `fix/single-rank-nccl-vram`
- **总耗时**: 82.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32266199885

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 46.5min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传失败产物时提示无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32266199885/job/96111728950) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32266199885/job/96111728950


## [Run #32265599205](https://github.com/sgl-project/sglang/actions/runs/32265599205)
- **分支**: `idhanani/unified-radix-swa-fix`
- **总耗时**: 119.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32265599205

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 63.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109539975) |
| base-b-test-2-npu-a3 / run (0) | 118.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540183) |
| base-b-test-4-npu-a3 / run (1) | 118.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540370) |
| base-b-test-8-npu-a3 / run (0) | 118.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540436) |
| base-b-test-4-npu-a3 / run (0) | 118.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540443) |
| base-b-test-1-npu-a3 / run (0) | 118.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540444) |
| base-b-test-16-npu-a3 / run (0) | 118.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540464) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 118.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540875) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 118.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109541047) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 118.0min | 环境问题 | Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109541053) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 118.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109541078) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109539975

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540183

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540370

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540436

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540443

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是构建产物或依赖未正确上传，需检查存储配置或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540444

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的构建产物或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540464

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540875

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109541047

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示BlobNotFound错误，说明CI作业依赖的某个文件或工件在Azure Blob存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109541053

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存未在存储账户中找到，可能是资源被清理、路径错误或上传失败，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109541078

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| check-changes | 0.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109249526) |
| base-a-test-1-npu-a2 / run (0) | 14.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32265599205/job/96109540596) |


## [Run #32265131842](https://github.com/sgl-project/sglang/actions/runs/32265131842)
- **分支**: `nemotron-3.5-spec-comparison`
- **总耗时**: 73.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32265131842

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 59.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833586) |
| base-b-test-4-npu-a3 / run (0) | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833610) |
| base-a-test-1-npu-a2 / run (0) | 4.6min | 环境问题 | NPU测试用例执行失败，返回未知应用异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833660) |
| base-b-test-8-npu-a3 / run (0) | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833664) |
| base-b-test-1-npu-a3 / run (0) | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833673) |
| base-b-test-2-npu-a3 / run (0) | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833730) |
| base-b-test-4-npu-a3 / run (1) | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833774) |
| base-b-test-16-npu-a3 / run (0) | 67.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833798) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107834181) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107834195) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107834242) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 67.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107834285) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时无文件，可能测试未运行或日志被截断，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833586

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833610

- **base-a-test-1-npu-a2 / run (0)**: test_npu_ascend_backend.py测试在NPU设备上运行时报ERR99999 UNKNOWN application exception，导致测试失败退出码1。可能是NPU环境配置或驱动问题，非代码逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833660

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833664

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833673

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833730

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 资源已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833774

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107833798

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107834181

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107834195

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107834242

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32265131842/job/96107834285


---
*Auto-generated by npu_pr_monitor.py*