# NPU CI 执行监控
**生成时间**: 2026-09-03 09:18 UTC
**分析 Run 数**: 27

---

## 📊 本次执行总结

- **成功 Job 数**: 23
- **失败 Run 数**: 27
- **成功 Job 平均耗时**: 13.1min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-a-test-1-npu-a2 / run (0) | 41.1min | #33714309567 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260521) |
| base-a-test-1-npu-a2 / run (0) | 32.8min | #33720465401 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486313) |
| base-b-test-4-npu-a3 / run (0) | 30.6min | #33713930713 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117751) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | #33713930713 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117786) |
| base-a-test-1-npu-a2 / run (0) | 19.5min | #33715109696 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727904) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | #33713930713 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117779) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | #33713930713 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117771) |
| base-a-test-1-npu-a2 / run (0) | 14.4min | #33716350370 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340984) |
| base-a-test-1-npu-a2 / run (0) | 13.8min | #33722992125 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980290) |
| base-a-test-1-npu-a2 / run (0) | 11.0min | #33721466638 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418625) |

### ❌ 耗时最长的失败 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| multimodal-gen-test-1-npu-a3 (0) | 278.5min | #33713930713 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117642) |
| multimodal-gen-test-1-npu-a3 (1) | 278.5min | #33713930713 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117645) |
| multimodal-gen-test-2-npu-a3 (0) | 278.5min | #33713930713 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117688) |
| multimodal-gen-test-2-npu-a3 (1) | 278.5min | #33713930713 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117712) |
| multimodal-gen-test-2-npu-a3 (1) | 212.8min | #33715109696 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727745) |
| multimodal-gen-test-1-npu-a3 (0) | 212.8min | #33715109696 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727798) |
| multimodal-gen-test-2-npu-a3 (0) | 212.8min | #33715109696 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727807) |
| base-b-test-4-npu-a3 / run (0) | 212.8min | #33715109696 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727859) |
| multimodal-gen-test-1-npu-a3 (1) | 212.8min | #33715109696 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727881) |
| base-b-test-4-npu-a3 / run (1) | 212.8min | #33715109696 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727885) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 | 失败任务链接 |
|----------|----------|------|---------|-------------|------|-------------|
| base-a-test-1-npu-a2 / run (0) | 23 | 18 | 1 | 0 | 4 | [job link 1](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694032) |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #33713930713<br>[#30775 Pipeline parallelism x speculative decoding (EAGLE/MTP) compatibility](https://github.com/sgl-project/sglang/pull/30775) | `feat/pp-spec-compat` | 279.4min | 5 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-16-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33713930713) |
| #33715109696<br>[#37424 [HiCache] Buffer mode support sidecar pool](https://github.com/sgl-project/sglang/pull/37424) | `support_buffer_mode_sidecar_pool` | 214.0min | 10 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), base-b-test-4-npu-a3 / run (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33715109696) |
| #33713337547<br>[#37133 [GLM-5.2] Keep GlmMoeDsa MoE e_score_correction_bias in fp32](https://github.com/sgl-project/sglang/pull/37133) | `rocm/glmmoedsa-correction-bias-fp32` | 169.6min | 10 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33713337547) |
| #33714309567<br>[#37654 [Model] Add native IFM K2 Horizon serving support](https://github.com/sgl-project/sglang/pull/37654) | `public/k2-horizon-runtime` | 169.0min | 10 | multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-1-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33714309567) |
| #33720465401 | `hicache-segment-lock-protocol` | 160.2min | 10 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33720465401) |
| #33713156786 | `perf/mtp-verify-attn-aiter-asm` | 120.9min | 10 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33713156786) |
| #33720398406 | `unified_kv_l3` | 103.5min | 10 | multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33720398406) |
| #33715333495 | `deepepv2-integration` | 60.6min | 10 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33715333495) |
| #33716153457<br>[#37503 [HiCache] L3 storage prefetch lifecycle metrics and cross-tier attribution fixes](https://github.com/sgl-project/sglang/pull/37503) | `feat/hicache-l3-io-query-metrics` | 59.7min | 10 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33716153457) |
| #33716474464 | `hicache-segment-lock-protocol` | 59.5min | 10 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33716474464) |
| #33715490569<br>[#37680 [diffusion] Plan residency as one pool on unified-memory devices](https://github.com/sgl-project/sglang/pull/37680) | `feat/planner-unified-memory` | 52.2min | 4 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33715490569) |
| #33718917056<br>[#37680 [diffusion] Plan residency as one pool on unified-memory devices](https://github.com/sgl-project/sglang/pull/37680) | `feat/planner-unified-memory` | 49.2min | 4 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33718917056) |
| #33716350370<br>[#30805 [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103](https://github.com/sgl-project/sglang/pull/30805) | `dsv4_fp8_trtllm_gen` | 35.9min | 10 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1), base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3 (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-2-npu-a3 (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33716350370) |
| #33713985480<br>[#37675 [Fix] Broadcast PP dynamic-chunk profiling failures so every rank disables together](https://github.com/sgl-project/sglang/pull/37675) | `main` | 33.3min | 10 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33713985480) |
| #33722475881<br>[#37680 [diffusion] Plan residency as one pool on unified-memory devices](https://github.com/sgl-project/sglang/pull/37680) | `feat/planner-unified-memory` | 31.5min | 4 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33722475881) |
| #33721484848<br>[#36349 [AMD][Diffusion] Migrate FlyDSL fused norm kernels to the v0.3.0 stable API](https://github.com/sgl-project/sglang/pull/36349) | `main` | 29.8min | 6 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33721484848) |
| #33721466638<br>[#37601 [AMD] support qlen>1 for aiter gluon path for Kimi K3](https://github.com/sgl-project/sglang/pull/37601) | `gluon_dspark` | 27.1min | 10 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33721466638) |
| #33714147927<br>[#37680 [diffusion] Plan residency as one pool on unified-memory devices](https://github.com/sgl-project/sglang/pull/37680) | `feat/planner-unified-memory` | 22.2min | 4 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33714147927) |
| #33721898138<br>[#37662 [diffusion] perf: fold the FastH3 VSA gate, fuse the H3 VAE qk-norm+RoPE under quality, scope the VSA-H3 recipe to the transformer](https://github.com/sgl-project/sglang/pull/37662) | `h3-perf` | 21.9min | 11 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), base-b-test-2-npu-a3 / run (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33721898138) |
| #33720348247<br>[#33723 [3/N] elastic-ep: Recapture decode CUDA graphs after scale-up](https://github.com/sgl-project/sglang/pull/33723) | `elastic-ep-cuda-graph-recapture` | 20.1min | 10 | multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), base-b-test-16-npu-a3 / run (0), multimodal-gen-test-1-npu-a3 (1), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33720348247) |
| #33722992125<br>[#35604 [CPU] Add native CPU kernel for MurmurHash32](https://github.com/sgl-project/sglang/pull/35604) | `cpu-murmur-hash` | 15.9min | 6 | base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33722992125) |
| #33713144329 | `public/k2-horizon-runtime` | 12.3min | 11 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33713144329) |
| #33721217004 | `mla_prefill_intel_xpu` | 8.9min | 10 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33721217004) |
| #33720948224<br>[#37118 [ROCm] Define the DSA head-gate graph helpers on HIP](https://github.com/sgl-project/sglang/pull/37118) | `main` | 7.7min | 11 | multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (0), base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33720948224) |
| #33713907554 | `public/k2-horizon-runtime` | 6.8min | 11 | multimodal-gen-test-2-npu-a3 (0), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (1), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33713907554) |
| #33715958111<br>[#30805 [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103](https://github.com/sgl-project/sglang/pull/30805) | `dsv4_fp8_trtllm_gen` | 6.7min | 10 | multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33715958111) |
| #33713639019<br>[#37675 [Fix] Broadcast PP dynamic-chunk profiling failures so every rank disables together](https://github.com/sgl-project/sglang/pull/37675) | `lsyin/pp-profile-failure-broadcast` | 6.1min | 11 | multimodal-gen-test-1-npu-a3 (0), multimodal-gen-test-1-npu-a3 (1), multimodal-gen-test-2-npu-a3 (1), multimodal-gen-test-2-npu-a3 (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0) | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/33713639019) |

---


## [Run #33722992125](https://github.com/sgl-project/sglang/actions/runs/33722992125)
- **分支**: `cpu-murmur-hash`
- **总耗时**: 15.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33722992125

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-16-npu-a3 / run (0) | 15.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980080) |
| base-b-test-8-npu-a3 / run (0) | 15.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980129) |
| base-b-test-4-npu-a3 / run (0) | 15.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980169) |
| base-b-test-4-npu-a3 / run (1) | 15.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980180) |
| base-b-test-1-npu-a3 / run (0) | 15.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980182) |
| base-b-test-2-npu-a3 / run (0) | 15.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980210) |

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980080

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980129

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查作业依赖的存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980169

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980180

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查相关 blob 路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980182

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980210

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 13.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33722992125/job/100545980290) |


## [Run #33722475881](https://github.com/sgl-project/sglang/actions/runs/33722475881)
- **分支**: `feat/planner-unified-memory`
- **总耗时**: 31.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33722475881

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 30.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722475881/job/100544422260) |
| multimodal-gen-test-1-npu-a3 (0) | 30.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722475881/job/100544422328) |
| multimodal-gen-test-2-npu-a3 (0) | 30.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722475881/job/100544422329) |
| multimodal-gen-test-1-npu-a3 (1) | 30.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33722475881/job/100544422349) |

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722475881/job/100544422260

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722475881/job/100544422328

- **multimodal-gen-test-2-npu-a3 (0)**: 作业日志返回BlobNotFound错误，说明CI流程尝试访问的Azure Blob存储资源缺失或路径错误，属于外部依赖环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722475881/job/100544422329

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33722475881/job/100544422349


## [Run #33721898138](https://github.com/sgl-project/sglang/actions/runs/33721898138)
- **分支**: `h3-perf`
- **总耗时**: 21.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33721898138

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 21.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693848) |
| multimodal-gen-test-2-npu-a3 (0) | 21.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693899) |
| multimodal-gen-test-1-npu-a3 (0) | 21.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693906) |
| base-b-test-2-npu-a3 / run (0) | 21.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693926) |
| multimodal-gen-test-1-npu-a3 (1) | 21.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693941) |
| base-b-test-8-npu-a3 / run (0) | 21.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693989) |
| base-b-test-1-npu-a3 / run (0) | 21.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694026) |
| base-a-test-1-npu-a2 / run (0) | 11.6min | 代码错误 | 测试文件缺少main入口导致收集测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694032) |
| base-b-test-4-npu-a3 / run (1) | 21.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694060) |
| base-b-test-16-npu-a3 / run (0) | 21.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694083) |
| base-b-test-4-npu-a3 / run (0) | 21.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694109) |

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob资源缺失，可能是日志或依赖文件未正确上传，或路径配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693848

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，属于环境配置或资源缺失问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693899

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693906

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传延迟、路径错误或文件被清理，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693926

- **multimodal-gen-test-1-npu-a3 (1)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693941

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查相关存储路径或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542693989

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694026

- **base-a-test-1-npu-a2 / run (0)**: test_vsa_block_sparse_sm100.py缺少`if __name__ == "__main__":`入口，pytest风格测试在`python3 file.py -f`下会静默跳过，需添加`sys.exit(pytest.main([__file__, "-v"]))`。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694032

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694060

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查相关资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694083

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721898138/job/100542694109


## [Run #33721484848](https://github.com/sgl-project/sglang/actions/runs/33721484848)
- **分支**: `main`
- **总耗时**: 29.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33721484848

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 28.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502686) |
| base-b-test-1-npu-a3 / run (0) | 28.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502726) |
| base-b-test-4-npu-a3 / run (0) | 28.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502753) |
| base-b-test-4-npu-a3 / run (1) | 28.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502757) |
| base-b-test-8-npu-a3 / run (0) | 28.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502871) |
| base-b-test-16-npu-a3 / run (0) | 28.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502888) |

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问某个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传延迟、路径错误或文件被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502686

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502726

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502753

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502757

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502871

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502888

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33721484848/job/100541502701) |


## [Run #33721466638](https://github.com/sgl-project/sglang/actions/runs/33721466638)
- **分支**: `gluon_dspark`
- **总耗时**: 27.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33721466638

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418326) |
| multimodal-gen-test-1-npu-a3 (1) | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418386) |
| base-b-test-2-npu-a3 / run (0) | 26.4min | 环境问题 | CI日志中引用的Azure Blob存储文件不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418442) |
| base-b-test-1-npu-a3 / run (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418454) |
| multimodal-gen-test-2-npu-a3 (1) | 26.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418459) |
| multimodal-gen-test-2-npu-a3 (0) | 26.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418497) |
| base-b-test-4-npu-a3 / run (1) | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418526) |
| base-b-test-8-npu-a3 / run (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418560) |
| base-b-test-4-npu-a3 / run (0) | 26.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418569) |
| base-b-test-16-npu-a3 / run (0) | 26.4min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418612) |

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）未上传或已被删除，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418326

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418386

- **base-b-test-2-npu-a3 / run (0)**: 作业日志返回BlobNotFound错误，表明构建或测试所需的预上传文件（如模型权重、缓存或日志）已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418442

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418454

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明作业尝试下载的工件或数据文件在存储账户中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查CI配置中的资源引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418459

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或存储被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418497

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418526

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418560

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418569

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该文件不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误所致，属于基础设施或环境问题，与代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418612

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 11.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33721466638/job/100541418625) |


## [Run #33721217004](https://github.com/sgl-project/sglang/actions/runs/33721217004)
- **分支**: `mla_prefill_intel_xpu`
- **总耗时**: 8.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33721217004

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540711912) |
| multimodal-gen-test-1-npu-a3 (0) | 8.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540711966) |
| multimodal-gen-test-2-npu-a3 (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540711971) |
| multimodal-gen-test-2-npu-a3 (1) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712003) |
| base-b-test-4-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712032) |
| base-b-test-16-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712074) |
| base-b-test-8-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712082) |
| base-b-test-1-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712083) |
| base-b-test-2-npu-a3 / run (0) | 8.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712110) |
| base-b-test-4-npu-a3 / run (1) | 8.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712153) |

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540711912

- **multimodal-gen-test-1-npu-a3 (0)**: 作业失败原因是访问Azure Blob存储时返回BlobNotFound错误，说明CI所需的模型权重、数据或日志文件未上传或路径错误，属于外部存储资源缺失的环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540711966

- **multimodal-gen-test-2-npu-a3 (0)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540711971

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712003

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712032

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关存储配置或重新触发作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712074

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712082

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在存储账户中已被删除或路径错误，属于基础设施或配置问题，需检查相关 blob 路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712083

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712110

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712153

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33721217004/job/100540712103) |


## [Run #33720948224](https://github.com/sgl-project/sglang/actions/runs/33720948224)
- **分支**: `main`
- **总耗时**: 7.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33720948224

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853193) |
| multimodal-gen-test-1-npu-a3 (1) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853205) |
| multimodal-gen-test-1-npu-a3 (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853242) |
| multimodal-gen-test-2-npu-a3 (0) | 7.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853282) |
| base-a-test-1-npu-a2 / run (0) | 6.8min | 环境问题 | 自定义容器执行失败，下载triton-ascend依赖时中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853335) |
| base-b-test-1-npu-a3 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853379) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853438) |
| base-b-test-2-npu-a3 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853455) |
| base-b-test-4-npu-a3 / run (1) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853464) |
| base-b-test-16-npu-a3 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853506) |
| base-b-test-4-npu-a3 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853618) |

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853193

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是存储配置错误或上游任务未成功生成文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853205

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853242

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853282

- **base-a-test-1-npu-a2 / run (0)**: 作业在安装triton-ascend==3.2.1.dev20260530时下载188.5MB的wheel包，下载过程中容器执行失败，提示联系自托管runner管理员，属于环境或网络问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853335

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853379

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853438

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853455

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是缓存或依赖文件未正确上传，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853464

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成产物，或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853506

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是上游产物未正确上传或过期清理所致，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720948224/job/100539853618


## [Run #33720465401](https://github.com/sgl-project/sglang/actions/runs/33720465401)
- **分支**: `hicache-segment-lock-protocol`
- **总耗时**: 160.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33720465401

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 159.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486072) |
| multimodal-gen-test-1-npu-a3 (0) | 159.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486085) |
| multimodal-gen-test-2-npu-a3 (0) | 159.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486093) |
| base-b-test-16-npu-a3 / run (0) | 159.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486103) |
| base-b-test-1-npu-a3 / run (0) | 159.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486151) |
| base-b-test-8-npu-a3 / run (0) | 159.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486199) |
| base-b-test-2-npu-a3 / run (0) | 159.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486226) |
| multimodal-gen-test-1-npu-a3 (1) | 159.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486248) |
| base-b-test-4-npu-a3 / run (0) | 159.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486267) |
| base-b-test-4-npu-a3 / run (1) | 159.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486316) |

- **multimodal-gen-test-2-npu-a3 (1)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486072

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失或配置错误，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486085

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486093

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486103

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件已被删除或路径错误，属于基础设施/存储配置问题，需检查相关 blob 的保留策略或上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486151

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、数据集或缓存）在 Azure Blob 中已被删除或路径错误，需检查资源是否存在或更新路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486199

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486226

- **multimodal-gen-test-1-npu-a3 (1)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志文件被清理、路径错误或上传失败，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486248

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486267

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486316

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 32.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33720465401/job/100538486313) |


## [Run #33720398406](https://github.com/sgl-project/sglang/actions/runs/33720398406)
- **分支**: `unified_kv_l3`
- **总耗时**: 103.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33720398406

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (0) | 102.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538375896) |
| multimodal-gen-test-1-npu-a3 (0) | 102.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538375918) |
| multimodal-gen-test-2-npu-a3 (1) | 102.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538375933) |
| multimodal-gen-test-1-npu-a3 (1) | 102.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538375980) |
| base-b-test-4-npu-a3 / run (1) | 102.4min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376025) |
| base-b-test-2-npu-a3 / run (0) | 102.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376078) |
| base-b-test-4-npu-a3 / run (0) | 102.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376090) |
| base-b-test-1-npu-a3 / run (0) | 102.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376100) |
| base-b-test-8-npu-a3 / run (0) | 102.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376103) |
| base-b-test-16-npu-a3 / run (0) | 102.4min | 环境问题 | Azure Blob存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376152) |

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538375896

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538375918

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失问题，非代码或性能原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538375933

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储对象已被删除或路径错误，可能是CI配置中引用的模型权重或数据文件未正确上传，需检查相关存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538375980

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是上游产物未上传或过期清理所致，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376025

- **base-b-test-2-npu-a3 / run (0)**: 作业失败原因是下载或访问Azure Blob存储中的文件时返回BlobNotFound错误，表明该blob已被删除或路径错误，属于外部存储资源缺失的环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376078

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376090

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成产物，或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376100

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376103

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个Azure Blob对象，但该对象已被删除或路径错误，返回BlobNotFound错误。这通常是CI配置中引用的工件或日志路径失效，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376152

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33720398406/job/100538376024) |


## [Run #33720348247](https://github.com/sgl-project/sglang/actions/runs/33720348247)
- **分支**: `elastic-ep-cuda-graph-recapture`
- **总耗时**: 20.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33720348247

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (0) | 19.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100537) |
| multimodal-gen-test-1-npu-a3 (0) | 19.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100578) |
| base-b-test-16-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100615) |
| multimodal-gen-test-1-npu-a3 (1) | 19.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100635) |
| base-b-test-4-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100662) |
| base-b-test-1-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100688) |
| multimodal-gen-test-2-npu-a3 (1) | 19.6min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100704) |
| base-b-test-4-npu-a3 / run (1) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100751) |
| base-b-test-2-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100810) |
| base-b-test-8-npu-a3 / run (0) | 19.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100818) |

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100537

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100578

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误导致，属于基础设施或环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100615

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的存储对象已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100635

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100662

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100688

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI依赖的远程存储对象缺失或路径错误，可能是上传失败、文件被清理或配置指向错误，属于基础设施/环境问题，需检查存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100704

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100751

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100810

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100818

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33720348247/job/100538100708) |


## [Run #33718917056](https://github.com/sgl-project/sglang/actions/runs/33718917056)
- **分支**: `feat/planner-unified-memory`
- **总耗时**: 49.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33718917056

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 48.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33718917056/job/100533927342) |
| multimodal-gen-test-1-npu-a3 (1) | 48.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33718917056/job/100533927398) |
| multimodal-gen-test-2-npu-a3 (1) | 48.7min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33718917056/job/100533927423) |
| multimodal-gen-test-2-npu-a3 (0) | 48.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33718917056/job/100533927438) |

- **multimodal-gen-test-1-npu-a3 (0)**: 作业尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施/存储问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33718917056/job/100533927342

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储资源缺失或路径错误，可能是日志上传或依赖文件未生成，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33718917056/job/100533927398

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33718917056/job/100533927423

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储账户中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33718917056/job/100533927438


## [Run #33716474464](https://github.com/sgl-project/sglang/actions/runs/33716474464)
- **分支**: `hicache-segment-lock-protocol`
- **总耗时**: 59.5min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33716474464

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 58.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679407) |
| multimodal-gen-test-2-npu-a3 (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679411) |
| multimodal-gen-test-1-npu-a3 (1) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679489) |
| multimodal-gen-test-1-npu-a3 (0) | 58.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679501) |
| base-b-test-16-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679602) |
| base-b-test-1-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679638) |
| base-b-test-4-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679674) |
| base-b-test-2-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679693) |
| base-b-test-4-npu-a3 / run (1) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679695) |
| base-b-test-8-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679751) |

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679407

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被删除或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679411

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679489

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或数据文件在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679501

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679602

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679638

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679674

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679693

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施/环境配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679695

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679751

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33716474464/job/100526679702) |


## [Run #33716350370](https://github.com/sgl-project/sglang/actions/runs/33716350370)
- **分支**: `dsv4_fp8_trtllm_gen`
- **总耗时**: 35.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33716350370

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 35.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340900) |
| multimodal-gen-test-1-npu-a3 (1) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340906) |
| base-b-test-2-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340974) |
| multimodal-gen-test-1-npu-a3 (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340979) |
| base-b-test-4-npu-a3 / run (1) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341017) |
| base-b-test-4-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341019) |
| base-b-test-8-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341030) |
| base-b-test-16-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341031) |
| base-b-test-1-npu-a3 / run (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341068) |
| multimodal-gen-test-2-npu-a3 (0) | 35.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341236) |

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob资源已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340900

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或测试数据）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340906

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340974

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340979

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341017

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查相关存储资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341019

- **base-b-test-8-npu-a3 / run (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341030

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是构建产物或缓存缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341031

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341068

- **multimodal-gen-test-2-npu-a3 (0)**: 作业尝试下载或访问一个不存在的 Blob 文件（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526341236

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33716350370/job/100526340984) |


## [Run #33716153457](https://github.com/sgl-project/sglang/actions/runs/33716153457)
- **分支**: `feat/hicache-l3-io-query-metrics`
- **总耗时**: 59.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33716153457

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765243) |
| multimodal-gen-test-2-npu-a3 (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765262) |
| multimodal-gen-test-1-npu-a3 (1) | 58.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765318) |
| base-b-test-1-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765326) |
| multimodal-gen-test-2-npu-a3 (1) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765334) |
| base-b-test-4-npu-a3 / run (1) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765345) |
| base-b-test-8-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765373) |
| base-b-test-16-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765382) |
| base-b-test-2-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765388) |
| base-b-test-4-npu-a3 / run (0) | 58.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765435) |

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765243

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765262

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765318

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765326

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765334

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765345

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765373

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765382

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传失败、路径错误或存储被清理所致，属于基础设施或环境配置问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765388

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 中已被删除或路径错误，需检查上传步骤或资源引用是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765435

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33716153457/job/100525765383) |


## [Run #33715958111](https://github.com/sgl-project/sglang/actions/runs/33715958111)
- **分支**: `dsv4_fp8_trtllm_gen`
- **总耗时**: 6.7min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33715958111

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 6.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173024) |
| multimodal-gen-test-1-npu-a3 (0) | 6.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173026) |
| multimodal-gen-test-2-npu-a3 (1) | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173063) |
| multimodal-gen-test-2-npu-a3 (0) | 6.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173070) |
| base-b-test-8-npu-a3 / run (0) | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173114) |
| base-b-test-4-npu-a3 / run (0) | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173127) |
| base-b-test-1-npu-a3 / run (0) | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173136) |
| base-b-test-16-npu-a3 / run (0) | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173142) |
| base-b-test-2-npu-a3 / run (0) | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173207) |
| base-b-test-4-npu-a3 / run (1) | 6.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173236) |

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储资源已被删除或路径错误，属于外部依赖环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173024

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖缺失或配置错误，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173026

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173063

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误、资源被清理或配置问题，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173070

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173114

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173127

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173136

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173142

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或已被删除，可能是上传失败或路径配置错误，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173207

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源已被删除或路径错误，属于环境或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173236

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33715958111/job/100525173208) |


## [Run #33715490569](https://github.com/sgl-project/sglang/actions/runs/33715490569)
- **分支**: `feat/planner-unified-memory`
- **总耗时**: 52.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33715490569

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 51.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715490569/job/100523785963) |
| multimodal-gen-test-2-npu-a3 (0) | 51.6min | 环境问题 | 日志下载失败，Blob不存在 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715490569/job/100523786011) |
| multimodal-gen-test-1-npu-a3 (0) | 51.6min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715490569/job/100523786061) |
| multimodal-gen-test-2-npu-a3 (1) | 51.6min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715490569/job/100523786085) |

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715490569/job/100523785963

- **multimodal-gen-test-2-npu-a3 (0)**: GitHub Actions作业日志指向的Azure Blob存储中不存在指定文件，可能是日志被清理或路径错误，导致无法获取实际测试输出，需检查日志存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715490569/job/100523786011

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715490569/job/100523786061

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的某个 blob 不存在，可能是日志上传失败、路径错误或存储被清理，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715490569/job/100523786085


## [Run #33715333495](https://github.com/sgl-project/sglang/actions/runs/33715333495)
- **分支**: `deepepv2-integration`
- **总耗时**: 60.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33715333495

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 59.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523336955) |
| multimodal-gen-test-2-npu-a3 (0) | 59.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523336970) |
| multimodal-gen-test-1-npu-a3 (0) | 59.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337065) |
| multimodal-gen-test-2-npu-a3 (1) | 59.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337083) |
| base-b-test-2-npu-a3 / run (0) | 59.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337088) |
| base-b-test-1-npu-a3 / run (0) | 59.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337134) |
| base-b-test-8-npu-a3 / run (0) | 59.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337161) |
| base-b-test-4-npu-a3 / run (1) | 59.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337179) |
| base-b-test-16-npu-a3 / run (0) | 59.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337216) |
| base-b-test-4-npu-a3 / run (0) | 59.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337235) |

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业依赖的某个文件（如模型权重或测试数据）在存储中缺失或路径错误，可能是上传失败、过期或被误删，需检查相关资源是否存在及路径配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523336955

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523336970

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337065

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置变更所致，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337083

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337088

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337134

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337161

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337179

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337216

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337235

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33715333495/job/100523337077) |


## [Run #33715109696](https://github.com/sgl-project/sglang/actions/runs/33715109696)
- **分支**: `support_buffer_mode_sidecar_pool`
- **总耗时**: 214.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33715109696

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 212.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727745) |
| multimodal-gen-test-1-npu-a3 (0) | 212.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727798) |
| multimodal-gen-test-2-npu-a3 (0) | 212.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，属于外部资源缺失。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727807) |
| base-b-test-4-npu-a3 / run (0) | 212.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727859) |
| multimodal-gen-test-1-npu-a3 (1) | 212.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727881) |
| base-b-test-4-npu-a3 / run (1) | 212.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727885) |
| base-b-test-1-npu-a3 / run (0) | 212.8min | 环境问题 | CI作业因依赖的Azure Blob存储文件不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727897) |
| base-b-test-2-npu-a3 / run (0) | 212.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727923) |
| base-b-test-16-npu-a3 / run (0) | 212.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727989) |
| base-b-test-8-npu-a3 / run (0) | 212.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727999) |

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，属于基础设施/资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727745

- **multimodal-gen-test-1-npu-a3 (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或存储被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727798

- **multimodal-gen-test-2-npu-a3 (0)**: 作业失败原因是下载或访问Azure Blob存储中的某个文件时返回BlobNotFound错误，即该文件不存在或路径错误。这通常由CI配置中的资源链接失效、文件被清理或上传失败导致，与代码或模型性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727807

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727859

- **multimodal-gen-test-1-npu-a3 (1)**: 作业尝试下载日志时，Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727881

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727885

- **base-b-test-1-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业尝试下载的预构建产物或缓存文件已被删除或路径错误，属于基础设施/环境配置问题，需检查CI脚本中的资源引用。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727897

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727923

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，可能是构建产物或依赖缓存缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727989

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727999

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 19.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33715109696/job/100522727904) |


## [Run #33714309567](https://github.com/sgl-project/sglang/actions/runs/33714309567)
- **分支**: `public/k2-horizon-runtime`
- **总耗时**: 169.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33714309567

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (0) | 168.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260342) |
| multimodal-gen-test-1-npu-a3 (1) | 168.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260346) |
| multimodal-gen-test-1-npu-a3 (0) | 168.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260403) |
| base-b-test-4-npu-a3 / run (1) | 168.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取所需数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260415) |
| base-b-test-8-npu-a3 / run (0) | 168.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260426) |
| base-b-test-16-npu-a3 / run (0) | 168.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260432) |
| base-b-test-2-npu-a3 / run (0) | 168.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260488) |
| base-b-test-4-npu-a3 / run (0) | 168.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260490) |
| multimodal-gen-test-2-npu-a3 (1) | 168.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260497) |
| base-b-test-1-npu-a3 / run (0) | 168.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260576) |

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260342

- **multimodal-gen-test-1-npu-a3 (1)**: 作业在尝试下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于外部依赖资源缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260346

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI依赖的远程存储对象缺失或路径错误，可能是上传失败、清理策略或配置问题，需检查作业依赖的工件或数据是否已正确生成并上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260403

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储中的某个 blob 不存在，可能是日志文件被清理、路径错误或上传失败，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260415

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260426

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260432

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该文件已被删除或路径错误，返回 BlobNotFound 错误。这通常是 CI 配置或存储生命周期问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260488

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是缓存清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260490

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260497

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，属于基础设施/存储配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260576

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33714309567/job/100520260521) |


## [Run #33714147927](https://github.com/sgl-project/sglang/actions/runs/33714147927)
- **分支**: `feat/planner-unified-memory`
- **总耗时**: 22.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33714147927

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 21.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714147927/job/100519716491) |
| multimodal-gen-test-1-npu-a3 (0) | 21.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714147927/job/100519716494) |
| multimodal-gen-test-2-npu-a3 (1) | 21.7min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取必要数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714147927/job/100519716515) |
| multimodal-gen-test-2-npu-a3 (0) | 21.7min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33714147927/job/100519716588) |

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714147927/job/100519716491

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI依赖的远程资源缺失或路径错误，可能是上传失败、文件被清理或配置指向了不存在的存储位置，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714147927/job/100519716494

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业依赖的某个日志或工件文件在 Azure Blob 存储中缺失或路径错误，可能是上传失败或清理策略导致，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714147927/job/100519716515

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI依赖的某个远程资源（如模型权重或测试数据）已被删除或路径错误，属于基础设施/环境配置问题，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33714147927/job/100519716588


## [Run #33713985480](https://github.com/sgl-project/sglang/actions/runs/33713985480)
- **分支**: `main`
- **总耗时**: 33.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33713985480

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (1) | 32.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260309) |
| multimodal-gen-test-2-npu-a3 (0) | 32.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260322) |
| multimodal-gen-test-1-npu-a3 (0) | 32.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260397) |
| multimodal-gen-test-1-npu-a3 (1) | 32.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260433) |
| base-b-test-2-npu-a3 / run (0) | 32.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260472) |
| base-b-test-4-npu-a3 / run (0) | 32.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260473) |
| base-b-test-1-npu-a3 / run (0) | 32.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260480) |
| base-b-test-4-npu-a3 / run (1) | 32.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260481) |
| base-b-test-8-npu-a3 / run (0) | 32.9min | 环境问题 | CI作业因Azure Blob存储中找不到指定文件而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260504) |
| base-b-test-16-npu-a3 / run (0) | 32.9min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260536) |

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明作业尝试下载的blob（可能为模型权重或测试数据）在存储账户中不存在，可能是资源被误删、路径配置错误或上传未完成，属于环境或资源准备问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260309

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 资源缺失或路径错误，可能是上传失败或配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260322

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或数据在 Azure Blob 存储中已被删除或路径错误，属于外部存储环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260397

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260433

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误所致，属于基础设施或环境问题，与代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260472

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260473

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，属于基础设施或配置问题，与代码逻辑无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260480

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260481

- **base-b-test-8-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业依赖的某个blob文件不存在或已被删除，可能是缓存或上传步骤异常，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260504

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260536

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 9.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713985480/job/100519260521) |


## [Run #33713930713](https://github.com/sgl-project/sglang/actions/runs/33713930713)
- **分支**: `feat/pp-spec-compat`
- **总耗时**: 279.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33713930713

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 278.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117642) |
| multimodal-gen-test-1-npu-a3 (1) | 278.5min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117645) |
| multimodal-gen-test-2-npu-a3 (0) | 278.5min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117688) |
| multimodal-gen-test-2-npu-a3 (1) | 278.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117712) |
| base-b-test-16-npu-a3 / run (0) | 13.2min | 环境问题 | 自定义容器执行失败，NPU作业在加载模型分片时中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117868) |

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117642

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117645

- **multimodal-gen-test-2-npu-a3 (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志上传失败、路径错误或文件被清理，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117688

- **multimodal-gen-test-2-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117712

- **base-b-test-16-npu-a3 / run (0)**: 日志显示作业在加载模型分片（约10%）时，自定义容器实现执行失败，提示联系自托管runner管理员，属于NPU环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117868

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-4-npu-a3 / run (0) | 30.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117751) |
| base-b-test-4-npu-a3 / run (1) | 15.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117771) |
| base-b-test-2-npu-a3 / run (0) | 18.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117779) |
| base-b-test-8-npu-a3 / run (0) | 7.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117781) |
| base-b-test-1-npu-a3 / run (0) | 24.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117786) |
| base-a-test-1-npu-a2 / run (0) | 9.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713930713/job/100519117792) |


## [Run #33713907554](https://github.com/sgl-project/sglang/actions/runs/33713907554)
- **分支**: `public/k2-horizon-runtime`
- **总耗时**: 6.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33713907554

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-2-npu-a3 (0) | 5.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080239) |
| multimodal-gen-test-1-npu-a3 (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080258) |
| multimodal-gen-test-1-npu-a3 (1) | 5.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080265) |
| multimodal-gen-test-2-npu-a3 (1) | 5.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080279) |
| base-b-test-2-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080313) |
| base-b-test-1-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080314) |
| base-a-test-1-npu-a2 / run (0) | 5.5min | 环境问题 | 自定义容器执行失败，NPU测试未实际运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080365) |
| base-b-test-16-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080430) |
| base-b-test-8-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080449) |
| base-b-test-4-npu-a3 / run (0) | 5.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080504) |
| base-b-test-4-npu-a3 / run (1) | 5.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080826) |

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明作业依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080239

- **multimodal-gen-test-1-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080258

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，属于外部依赖资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080265

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储账户中已被删除或路径错误，属于基础设施/环境配置问题，需检查上传步骤或存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080279

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080313

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080314

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动测试前因自定义容器实现执行失败而中止，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非测试代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080365

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重、缓存或构建产物）已被删除或路径错误，需检查上传步骤或资源生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080430

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件未上传或已被删除，可能是上游任务未成功生成或存储配置有误。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080449

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080504

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713907554/job/100519080826


## [Run #33713639019](https://github.com/sgl-project/sglang/actions/runs/33713639019)
- **分支**: `lsyin/pp-profile-failure-broadcast`
- **总耗时**: 6.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33713639019

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 5.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250383) |
| multimodal-gen-test-1-npu-a3 (1) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250408) |
| multimodal-gen-test-2-npu-a3 (1) | 5.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250410) |
| multimodal-gen-test-2-npu-a3 (0) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250418) |
| base-b-test-4-npu-a3 / run (1) | 5.2min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250563) |
| base-b-test-1-npu-a3 / run (0) | 5.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250567) |
| base-a-test-1-npu-a2 / run (0) | 4.5min | 环境问题 | 下载triton-ascend依赖时网络中断导致容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250571) |
| base-b-test-4-npu-a3 / run (0) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250606) |
| base-b-test-8-npu-a3 / run (0) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250615) |
| base-b-test-16-npu-a3 / run (0) | 5.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250636) |
| base-b-test-2-npu-a3 / run (0) | 5.2min | 环境问题 | CI作业因Azure Blob存储中指定文件不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250701) |

- **multimodal-gen-test-1-npu-a3 (0)**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于外部存储资源缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250383

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250408

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob资源缺失，可能是日志上传或下载路径错误，或资源已被删除。属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250410

- **multimodal-gen-test-2-npu-a3 (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250418

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是上游产物未上传或过期，需检查依赖的构建产物是否成功生成。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250563

- **base-b-test-1-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储账户中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250567

- **base-a-test-1-npu-a2 / run (0)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的wheel包过程中网络连接中断，仅下载5.2MB后重试仍失败，最终导致自定义容器执行报错，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250571

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250606

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250615

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250636

- **base-b-test-2-npu-a3 / run (0)**: 日志显示BlobNotFound错误，说明作业尝试下载的构建产物或依赖文件在存储中缺失，可能是上游作业未成功上传或文件被清理，属于环境/基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713639019/job/100518250701


## [Run #33713337547](https://github.com/sgl-project/sglang/actions/runs/33713337547)
- **分支**: `rocm/glmmoedsa-correction-bias-fp32`
- **总耗时**: 169.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33713337547

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 168.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390036) |
| multimodal-gen-test-2-npu-a3 (0) | 168.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390042) |
| multimodal-gen-test-1-npu-a3 (0) | 168.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390044) |
| multimodal-gen-test-2-npu-a3 (1) | 168.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390062) |
| base-b-test-1-npu-a3 / run (0) | 168.8min | 环境问题 | CI作业因Azure Blob存储中指定的blob不存在而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390094) |
| base-b-test-2-npu-a3 / run (0) | 168.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390161) |
| base-b-test-16-npu-a3 / run (0) | 168.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390171) |
| base-b-test-4-npu-a3 / run (0) | 168.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390207) |
| base-b-test-4-npu-a3 / run (1) | 168.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390252) |
| base-b-test-8-npu-a3 / run (0) | 168.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390338) |

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390036

- **multimodal-gen-test-2-npu-a3 (0)**: 作业尝试下载或访问的 blob 资源未找到（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于外部存储环境问题，非代码或性能原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390042

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390044

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或存储被清理，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390062

- **base-b-test-1-npu-a3 / run (0)**: 日志显示BlobNotFound错误，表明作业尝试下载或访问的构建产物/缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390094

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390161

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390171

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390207

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是缓存清理或配置问题，需检查相关资源是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390252

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 已被删除或路径错误，属于基础设施或配置问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390338

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713337547/job/100517390213) |


## [Run #33713156786](https://github.com/sgl-project/sglang/actions/runs/33713156786)
- **分支**: `perf/mtp-verify-attn-aiter-asm`
- **总耗时**: 120.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33713156786

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (0) | 120.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815257) |
| multimodal-gen-test-2-npu-a3 (1) | 120.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815272) |
| multimodal-gen-test-1-npu-a3 (1) | 120.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815280) |
| multimodal-gen-test-2-npu-a3 (0) | 120.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取数据。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815307) |
| base-b-test-1-npu-a3 / run (0) | 120.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815337) |
| base-b-test-8-npu-a3 / run (0) | 120.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815357) |
| base-b-test-4-npu-a3 / run (1) | 120.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815408) |
| base-b-test-2-npu-a3 / run (0) | 120.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815413) |
| base-b-test-16-npu-a3 / run (0) | 120.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815415) |
| base-b-test-4-npu-a3 / run (0) | 120.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815438) |

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试访问的Azure Blob存储对象已被删除或路径错误，可能是CI配置引用了不存在的工件或缓存，需检查相关存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815257

- **multimodal-gen-test-2-npu-a3 (1)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815272

- **multimodal-gen-test-1-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob文件已被删除或路径错误，属于外部存储资源缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815280

- **multimodal-gen-test-2-npu-a3 (0)**: 作业尝试下载或访问一个不存在的 Azure Blob 对象（BlobNotFound），可能是日志文件被清理、路径错误或上传失败，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815307

- **base-b-test-1-npu-a3 / run (0)**: 作业尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。可能是日志上传失败、路径错误或文件被清理，属于基础设施/环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815337

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815357

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，需检查相关 blob 的路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815408

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是构建产物或依赖未正确上传，需检查存储配置或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815413

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或已被删除，可能是资源清理或路径配置错误，需检查相关存储路径和文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815415

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查资源上传或引用配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815438

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/33713156786/job/100516815467) |


## [Run #33713144329](https://github.com/sgl-project/sglang/actions/runs/33713144329)
- **分支**: `public/k2-horizon-runtime`
- **总耗时**: 12.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/33713144329

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 (1) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828131) |
| multimodal-gen-test-1-npu-a3 (0) | 11.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828139) |
| multimodal-gen-test-2-npu-a3 (1) | 11.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828159) |
| multimodal-gen-test-2-npu-a3 (0) | 11.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828164) |
| base-b-test-2-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828243) |
| base-b-test-8-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828298) |
| base-b-test-1-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828316) |
| base-b-test-4-npu-a3 / run (1) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828325) |
| base-a-test-1-npu-a2 / run (0) | 11.0min | 其他 | 作业实际测试全部通过，但日志显示作业被标记为失败，可能因后续清理或基础设施问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828340) |
| base-b-test-4-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828385) |
| base-b-test-16-npu-a3 / run (0) | 11.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828442) |

- **multimodal-gen-test-1-npu-a3 (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，属于外部存储环境问题，需检查资源是否存在或更新路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828131

- **multimodal-gen-test-1-npu-a3 (0)**: 错误码BlobNotFound表明作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828139

- **multimodal-gen-test-2-npu-a3 (1)**: 错误码BlobNotFound表明CI作业尝试下载的依赖文件或缓存已从存储中删除或路径错误，属于基础设施配置或资源缺失问题，需检查作业引用的blob路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828159

- **multimodal-gen-test-2-npu-a3 (0)**: 错误码BlobNotFound表明CI依赖的远程存储对象缺失或路径错误，可能是上传失败、清理策略或配置变更所致，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828164

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure 存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828243

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件已被删除或路径错误，属于基础设施/存储配置问题，需检查相关 blob 的保留策略或上传步骤。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828298

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828316

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径及生命周期策略。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828325

- **base-a-test-1-npu-a2 / run (0)**: 测试摘要显示2/2通过，无测试失败。失败可能源于作业后清理阶段或基础设施异常，如Node 20弃用警告或runner环境问题，但日志未显示明确错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828340

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828385

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在存储中已被删除或路径错误，属于基础设施/环境配置问题，需检查相关 blob 路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/33713144329/job/100516828442


---
*Auto-generated by npu_pr_monitor.py*