# NPU CI 执行监控
**生成时间**: 2026-08-15 08:12 UTC
**分析 Run 数**: 21

---

## 📊 本次执行总结

- **成功 Job 数**: 114
- **失败 Run 数**: 20
- **成功 Job 平均耗时**: 28.8min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 275.2min | #31602306929 | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94153379410) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 129.2min | #31602306929 | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678968) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.5min | #31586352481 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081085356) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 102.0min | #31598793346 | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120839079) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.9min | #31592005298 | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099001211) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.2min | #31585698126 | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080307049) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.4min | #31609875181 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159985552) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.4min | #31592005298 | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94158134212) |
| base-b-test-16-npu-a3 / run (0) | 63.3min | #31602306929 | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678282) |
| base-b-test-16-npu-a3 / run (0) | 55.5min | #31586352481 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081084937) |

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败任务 | 结论 | 链接 |
|--------|------|------|-----------|---------|------|------|
| #31602306929<br>[#33089 [NPU] Add sparsity-driven KV offload for DeepSeek DSA on Ascend](https://github.com/sgl-project/sglang/pull/33089) | `tcj/sparsity-driven-kv-offload` | 343.6min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31602306929) |
| #31592005298<br>[#28932 [AMD] Add dense-FP8 for MXFP4 checkpoints with fused silu, mul, activation quant](https://github.com/sgl-project/sglang/pull/28932) | `marv/fuse_down_proj_act_quant_silu_mul` | 288.1min | 2 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31592005298) |
| #31586352481<br>[#32598 [UT][NPU] Add npu unit test for ascend_gdn_backend and ascend_hybrid_linear_attn_backend](https://github.com/sgl-project/sglang/pull/32598) | `dev` | 281.9min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31586352481) |
| #31588237155<br>[#34275 [Diffusion] Fuse Cosmos3 QK norm, RoPE, and KV packing](https://github.com/sgl-project/sglang/pull/34275) | `codex/cosmos3-qknorm-rope-fusion-pr` | 281.6min | 4 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31588237155) |
| #31585698126<br>[#28354 [FlashInfer v0.6.16] Support FlashInfer CuTe DSL NVFP4 MoE quantization](https://github.com/sgl-project/sglang/pull/28354) | `agent-cutedsl-moe-nvfp4-sglang` | 263.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31585698126) |
| #31586416120 | `feature/load-reporter` | 200.6min | 11 | base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31586416120) |
| #31598793346 | `fuse-swiglu-moe-up-gemm-epilogue` | 159.6min | 4 | base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31598793346) |
| #31607912862<br>[#33561 [Model] Support Ling-3.0-flash (BailingMoeV3) ](https://github.com/sgl-project/sglang/pull/33561) | `ling3-flash-dspark` | 128.4min | 4 | base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31607912862) |
| #31600895510<br>[#33465 [Kimi-K3][NPU]  Support Kimi-K3 on NPU](https://github.com/sgl-project/sglang/pull/33465) | `main` | 108.1min | 6 | base-b-test-4-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31600895510) |
| #31594876012<br>[#34411 [VLM] Reuse cached Kimi-K3 embeddings before preprocessing](https://github.com/sgl-project/sglang/pull/34411) | `codex/k3-mm-cache-lease` | 103.7min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31594876012) |
| #31609875181<br>[#34584 [diffusion] Wan2.2-TI2V: fuse per-token adaLN table add into contiguous slices + hoist rope cache (denoise -13.1% H100 / -12.6% H200, bit-exact; eager beats compile)](https://github.com/sgl-project/sglang/pull/34584) | `diffusion-wan-ti2v-modulation-fusion` | 94.3min | 5 | base-b-test-4-npu-a3 / run (0), base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3, base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3, base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3, base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/31609875181) |
| #31590137211<br>[#34411 [VLM] Reuse cached Kimi-K3 embeddings before preprocessing](https://github.com/sgl-project/sglang/pull/34411) | `codex/k3-mm-cache-lease` | 63.1min | 10 | base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31590137211) |
| #31604903070<br>[#34042 add flashinfer cute-dsl backend for mxfp8 gemm](https://github.com/sgl-project/sglang/pull/34042) | `feat/mxfp8-cutedsl` | 24.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31604903070) |
| #31606260835<br>[#33561 [Model] Support Ling-3.0-flash (BailingMoeV3) ](https://github.com/sgl-project/sglang/pull/33561) | `ling3-flash-dspark` | 18.9min | 11 | multimodal-gen-test-1-npu-a3, base-a-test-1-npu-a2 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31606260835) |
| #31586629612<br>[#31575 Fix rope config compatibility and VL/transformers-fallback weight loading](https://github.com/sgl-project/sglang/pull/31575) | `fix/rope-config-and-vl-weight-loading` | 12.8min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31586629612) |
| #31588931683<br>[#34411 [VLM] Reuse cached Kimi-K3 embeddings before preprocessing](https://github.com/sgl-project/sglang/pull/34411) | `codex/k3-mm-cache-lease` | 10.9min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31588931683) |
| #31587676877<br>[#34411 [VLM] Reuse cached Kimi-K3 embeddings before preprocessing](https://github.com/sgl-project/sglang/pull/34411) | `codex/k3-mm-cache-lease` | 9.3min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31587676877) |
| #31588342274<br>[#34411 [VLM] Reuse cached Kimi-K3 embeddings before preprocessing](https://github.com/sgl-project/sglang/pull/34411) | `codex/k3-mm-cache-lease` | 8.2min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31588342274) |
| #31589699557<br>[#34411 [VLM] Reuse cached Kimi-K3 embeddings before preprocessing](https://github.com/sgl-project/sglang/pull/34411) | `codex/k3-mm-cache-lease` | 6.4min | 12 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-a-test-1-npu-a2 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31589699557) |
| #31609825263 | `diffusion-wan-ti2v-modulation-fusion` | 5.8min | 3 | base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/31609825263) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 失败 | 取消 |
|----------|----------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 19 | 13 | 0 | 6 |
| base-b-test-1-npu-a3 / run (0) | 20 | 9 | 0 | 11 |
| base-b-test-16-npu-a3 / run (0) | 19 | 8 | 0 | 11 |
| base-b-test-2-npu-a3 / run (0) | 20 | 9 | 0 | 11 |
| base-b-test-4-npu-a3 / run (0) | 19 | 7 | 0 | 12 |
| base-b-test-4-npu-a3 / run (1) | 19 | 9 | 0 | 10 |
| base-b-test-8-npu-a3 / run (0) | 19 | 10 | 0 | 9 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19 | 9 | 0 | 10 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 20 | 6 | 0 | 14 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 19 | 9 | 0 | 10 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19 | 10 | 0 | 9 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 9 | 1 | 0 | 8 |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 6 | 1 | 0 | 5 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 9 | 2 | 0 | 7 |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 10 | 0 | 0 | 10 |
| multimodal-gen-test-1-npu-a3 | 19 | 11 | 1 | 7 |

---


## [Run #31609875181](https://github.com/sgl-project/sglang/actions/runs/31609875181)
- **分支**: `diffusion-wan-ti2v-modulation-fusion`
- **总耗时**: 94.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31609875181

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 19.8min | 代码错误 | NPU DP注意力测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984790) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.0min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94161452245) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94167408258) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94171915752) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94185312675) |

- **base-b-test-4-npu-a3 / run (0)**: test_npu_dp_attention.py测试返回退出码1，5个测试中仅1个通过，该测试耗时762秒后失败，可能涉及DP注意力功能实现问题或环境配置错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984790

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1117秒后失败，0/1通过，属于性能测试未达标，可能是模型推理性能未满足预设阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94161452245

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到 base-b-test-4-npu-a3 / run (0) 作业失败，被判定为根因失败，因此本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94167408258

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 作业在启动阶段执行PR健康检查时，检测到base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3两个根因作业已失败，因此本作业被快速失败机制跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94171915752

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现根因失败作业（base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3），本作业作为级联失败被快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94185312675

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 26.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984594) |
| base-b-test-2-npu-a3 / run (0) | 21.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984656) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984705) |
| base-b-test-16-npu-a3 / run (0) | 46.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984778) |
| base-b-test-4-npu-a3 / run (1) | 16.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984794) |
| base-b-test-1-npu-a3 / run (0) | 24.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984894) |
| base-a-test-1-npu-a2 / run (0) | 8.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159984904) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159985467) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 38.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159985473) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159985533) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 84.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31609875181/job/94159985552) |


## [Run #31609825263](https://github.com/sgl-project/sglang/actions/runs/31609825263)
- **分支**: `diffusion-wan-ti2v-modulation-fusion`
- **总耗时**: 5.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31609825263

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-2-npu-a3 / run (0) | 0.8min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609825263/job/94158148783) |
| base-b-test-1-npu-a3 / run (0) | 5.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609825263/job/94158148930) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 1.0min | 环境问题 | 自定义容器启动失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31609825263/job/94158149198) |

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动自定义容器时失败，错误信息为"Executing the custom container implementation failed"，提示联系自托管runner管理员，属于runner环境配置或容器启动问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31609825263/job/94158148783

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31609825263/job/94158148930

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示在下载依赖包时，执行自定义容器实现失败，错误信息为“Executing the custom container implementation failed”，提示联系自托管 runner 管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31609825263/job/94158149198


## [Run #31607912862](https://github.com/sgl-project/sglang/actions/runs/31607912862)
- **分支**: `ling3-flash-dspark`
- **总耗时**: 128.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31607912862

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 125.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151989049) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94153217253) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败导致快速失败，本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94160117724) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94165608737) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在执行过程中出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151989049

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行失败，耗时1099秒，未通过性能指标要求，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94153217253

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业未实际执行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94160117724

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在健康检查阶段因检测到根因作业base-c-test-perf-8-npu-a3失败而触发快速失败机制，本作业被跳过未实际运行。
  链接: https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94165608737

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151988052) |
| base-a-test-1-npu-a2 / run (0) | 4.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151988177) |
| base-b-test-1-npu-a3 / run (0) | 22.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151988266) |
| base-b-test-4-npu-a3 / run (0) | 29.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151988271) |
| base-b-test-2-npu-a3 / run (0) | 20.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151988378) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151988413) |
| base-b-test-4-npu-a3 / run (1) | 16.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151988516) |
| base-b-test-16-npu-a3 / run (0) | 47.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151988560) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151989046) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151989062) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 25.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31607912862/job/94151989347) |


## [Run #31606260835](https://github.com/sgl-project/sglang/actions/runs/31606260835)
- **分支**: `ling3-flash-dspark`
- **总耗时**: 18.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31606260835

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 17.9min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145993285) |
| base-a-test-1-npu-a2 / run (0) | 16.6min | 环境问题 | 自托管runner执行自定义容器实现失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145993941) |
| base-b-test-1-npu-a3 / run (0) | 11.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994080) |
| base-b-test-16-npu-a3 / run (0) | 9.2min | 环境问题 | 自定义容器执行失败，NPU测试环境异常中断 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994152) |
| base-b-test-4-npu-a3 / run (0) | 14.5min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994170) |
| base-b-test-4-npu-a3 / run (1) | 5.7min | 环境问题 | 自定义容器执行失败，导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994321) |
| base-b-test-2-npu-a3 / run (0) | 10.6min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994332) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 10.1min | 环境问题 | 自定义容器执行失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994931) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.2min | 环境问题 | 自定义容器执行失败，导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994940) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145995015) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 8.1min | 环境问题 | 自定义容器执行失败，NPU环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94148917015) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未生成失败产物，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145993285

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在下载依赖包过程中，runner报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145993941

- **base-b-test-1-npu-a3 / run (0)**: 作业在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员。日志显示NPU内存正常但容器中途崩溃，属于环境配置或容器兼容性问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994080

- **base-b-test-16-npu-a3 / run (0)**: 日志显示服务启动正常，但随后出现"Executing the custom container implementation failed"错误，提示联系self-hosted runner管理员，属于NPU容器环境问题导致作业失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994152

- **base-b-test-4-npu-a3 / run (0)**: 作业在加载模型权重时（Multi-thread loading shards 0%）容器实现执行失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试逻辑错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994170

- **base-b-test-4-npu-a3 / run (1)**: 日志显示在加载模型权重时，自定义容器实现执行失败，提示联系自托管runner管理员，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994321

- **base-b-test-2-npu-a3 / run (0)**: 日志显示容器启动后加载模型时出现导入错误（如缺少vllm、mindspore模块），随后自定义容器实现执行失败，导致作业终止。这属于NPU测试环境配置或依赖缺失问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994332

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示在安装evalscope依赖时，自定义容器实现执行失败（Executing the custom container implementation failed），可能是容器环境或镜像问题，需联系自托管runner管理员排查。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994931

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在捕获批次过程中，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994940

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试请求均返回200，但随后出现"Executing the custom container implementation failed"错误，表明runner在执行自定义容器时环境故障，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145995015

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示模型分片加载至79%时，GitHub Actions报错“Executing the custom container implementation failed”，提示联系自托管runner管理员，属于NPU容器环境问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94148917015

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145993819) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31606260835/job/94145994927) |


## [Run #31604903070](https://github.com/sgl-project/sglang/actions/runs/31604903070)
- **分支**: `feat/mxfp8-cutedsl`
- **总耗时**: 24.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31604903070

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.4min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380069) |
| base-b-test-2-npu-a3 / run (0) | 19.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380119) |
| base-b-test-16-npu-a3 / run (0) | 10.0min | 环境问题 | 自定义容器执行失败，可能是NPU环境或容器配置问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380125) |
| base-b-test-8-npu-a3 / run (0) | 19.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380230) |
| base-b-test-4-npu-a3 / run (0) | 1.7min | 环境问题 | 自托管runner在下载triton-ascend依赖时容器执行失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380349) |
| base-b-test-1-npu-a3 / run (0) | 19.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380366) |
| base-b-test-4-npu-a3 / run (1) | 3.2min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380422) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 19.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380685) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 3.4min | 环境问题 | 自定义容器启动失败，导致作业在依赖安装阶段中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380758) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 19.5min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380759) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 19.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380804) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380069

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的日志 blob 已被删除或路径错误，可能是日志上传失败或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380119

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动自定义容器时失败，错误提示为执行自定义容器实现失败，需联系自托管runner管理员。可能是NPU驱动、容器镜像或资源分配问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380125

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380230

- **base-b-test-4-npu-a3 / run (0)**: 作业在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中，自定义容器实现执行失败，导致作业中断。属于runner环境或容器配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380349

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于外部依赖缺失，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380366

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动测试前，执行自定义容器实现时失败，错误提示需联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身导致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380422

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380685

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示在安装triton-ascend等依赖时，自定义容器实现执行失败，错误为'Executing the custom container implementation failed'，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380758

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的 blob 文件缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380759

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380804

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31604903070/job/94141380247) |


## [Run #31602306929](https://github.com/sgl-project/sglang/actions/runs/31602306929)
- **分支**: `tcj/sparsity-driven-kv-offload`
- **总耗时**: 343.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31602306929

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.2min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94147133973) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94180638141) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1122秒后失败，0/1通过，属于性能测试未达标，可能因模型推理速度或延迟不满足50ms要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94147133973

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示健康检查发现base-c-test-perf-8-npu-a3作业失败，被判定为根因失败，因此本作业（base-c-test-perf-2-npu-a3）被快速失败跳过，并非自身执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94180638141

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678004) |
| base-b-test-16-npu-a3 / run (0) | 63.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678282) |
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678327) |
| base-b-test-8-npu-a3 / run (0) | 9.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678366) |
| base-b-test-4-npu-a3 / run (0) | 33.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678376) |
| base-b-test-1-npu-a3 / run (0) | 22.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678397) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678431) |
| base-b-test-4-npu-a3 / run (1) | 15.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678524) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 36.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678908) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 27.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678925) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678952) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 129.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94132678968) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94151532911) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 275.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31602306929/job/94153379410) |


## [Run #31600895510](https://github.com/sgl-project/sglang/actions/runs/31600895510)
- **分支**: `main`
- **总耗时**: 108.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31600895510

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-4-npu-a3 / run (0) | 8.9min | 超时 | NPU测试用例test_npu_hicache_mla.py执行超时（301秒）导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833182) |
| base-b-test-16-npu-a3 / run (0) | 74.1min | 其他 | 作业实际成功，所有测试通过，无失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833184) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 69.7min | 精度回归 | NPU精度测试用例qwen3_5_9b_bf16_1p_gsm8k失败，0/3测试通过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833469) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 23.3min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94138713255) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 4.1min | 其他 | 作业被健康检查快速失败机制跳过，因其他根因作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94144732604) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 作业被健康检查快速失败机制跳过，因其他根因作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94148942819) |

- **base-b-test-4-npu-a3 / run (0)**: 测试文件test/registered/npu/basic_function/HiCache/test_npu_hicache_mla.py运行超过300秒超时限制，返回退出码1，最终导致整个作业失败。日志显示该文件是唯一失败的测试，且耗时301秒。
  链接: https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833182

- **base-b-test-16-npu-a3 / run (0)**: 日志显示所有6个NPU测试文件均通过（passed: true），作业正常完成清理流程，仅有Node 20弃用警告，无任何错误或失败迹象。
  链接: https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833184

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 测试test_npu_qwen3_5_9b_bf16_1p_gsm8k.py返回退出码1，耗时3954秒超过预估3600秒，所有3个测试均未通过，属于精度回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833469

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1192秒后退出码1，属于性能测试未通过，可能因吞吐或延迟未达预期阈值。
  链接: https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94138713255

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 该作业本身未执行测试，因健康检查检测到base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3两个根因作业失败，触发了fast-fail跳过机制，导致本作业被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94144732604

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业未实际运行，在健康检查阶段因检测到其他根因作业（base-b-test-4-npu-a3和base-c-test-perf-8-npu-a3）失败而被快速失败跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94148942819

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 36.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127832923) |
| base-b-test-1-npu-a3 / run (0) | 47.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833095) |
| base-b-test-4-npu-a3 / run (1) | 27.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833118) |
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833156) |
| base-b-test-8-npu-a3 / run (0) | 11.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833242) |
| base-b-test-2-npu-a3 / run (0) | 28.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833312) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833435) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 37.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833463) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31600895510/job/94127833568) |


## [Run #31598793346](https://github.com/sgl-project/sglang/actions/runs/31598793346)
- **分支**: `fuse-swiglu-moe-up-gemm-epilogue`
- **总耗时**: 159.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31598793346

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.7min | 性能回归 | NPU性能测试未通过，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94137854372) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查快速失败，因同PR中另一作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94145209570) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94148785860) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业（perf-8-npu-a3）已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94168649873) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1099秒后退出码为1，0/1测试通过，属于性能测试未达标或执行失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94137854372

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到同PR的base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于CI依赖失败而非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94145209570

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动阶段被健康检查快速失败机制终止，原因是同PR中base-c-test-perf-8-npu-a3作业失败，本作业被判定为级联失败而跳过，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94148785860

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 本作业在启动前的PR健康检查阶段检测到根因作业base-c-test-perf-8-npu-a3失败，触发fast-fail机制，本作业未实际运行即被终止，属于级联跳过而非自身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94168649873

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 30.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120838742) |
| base-a-test-1-npu-a2 / run (0) | 11.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120838773) |
| base-b-test-2-npu-a3 / run (0) | 20.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120838796) |
| base-b-test-4-npu-a3 / run (0) | 30.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120838817) |
| base-b-test-1-npu-a3 / run (0) | 22.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120838886) |
| base-b-test-4-npu-a3 / run (1) | 14.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120838913) |
| base-b-test-8-npu-a3 / run (0) | 8.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120838986) |
| base-b-test-16-npu-a3 / run (0) | 46.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120838999) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 102.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120839079) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120839107) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120839178) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 28.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31598793346/job/94120839222) |


## [Run #31594876012](https://github.com/sgl-project/sglang/actions/runs/31594876012)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 103.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31594876012

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 39.7min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024095) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024267) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024300) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联取消。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024330) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024382) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查快速失败，根因作业为multimodal-gen-test-1-npu-a3 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024407) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024433) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.3min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024666) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024794) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024836) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024837) |

- **multimodal-gen-test-1-npu-a3**: 日志仅包含GitHub Actions运行器初始化、Node版本警告及上传artifact步骤，未显示multimodal-gen测试的具体执行结果或错误信息，无法判断失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024095

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动前的健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024267

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024300

- **base-b-test-4-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而触发fast-fail跳过，并非自身问题，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024330

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024382

- **base-b-test-16-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而触发快速失败机制，并非自身测试失败，属于级联取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024407

- **base-b-test-2-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非自身测试失败，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024433

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤并快速失败，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024666

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024794

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于CI依赖链问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024836

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024837

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31594876012/job/94108024320) |


## [Run #31594673891](https://github.com/sgl-project/sglang/actions/runs/31594673891)
- **分支**: `main`
- **总耗时**: 38.3min | **结论**: success
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31594673891

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 34.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31594673891/job/94107400392) |


## [Run #31592005298](https://github.com/sgl-project/sglang/actions/runs/31592005298)
- **分支**: `marv/fuse_down_proj_act_quant_silu_mul`
- **总耗时**: 288.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31592005298

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.6min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94130466001) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 32.9min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94139010129) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试文件test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1099秒后返回退出码1，0/1通过，表明性能指标未达标或执行出错，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94130466001

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试test_npu_kimi_k2_6_w4a8_8p_in3k5_out1k5_20ms.py执行1664秒后失败，退出码1，4个测试全部未通过，属于性能指标未达到预期要求。
  链接: https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94139010129

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 27.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099000862) |
| base-b-test-2-npu-a3 / run (0) | 18.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099000935) |
| base-b-test-8-npu-a3 / run (0) | 7.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099000948) |
| base-b-test-16-npu-a3 / run (0) | 46.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099000986) |
| base-b-test-4-npu-a3 / run (1) | 13.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099000997) |
| base-b-test-4-npu-a3 / run (0) | 29.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099001034) |
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099001082) |
| base-b-test-1-npu-a3 / run (0) | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099001191) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 87.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099001211) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099001324) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099001365) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94099001414) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 19.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94144039912) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 76.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31592005298/job/94158134212) |


## [Run #31590137211](https://github.com/sgl-project/sglang/actions/runs/31590137211)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 63.1min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31590137211

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 62.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140582) |
| base-b-test-1-npu-a3 / run (0) | 62.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140614) |
| base-b-test-2-npu-a3 / run (0) | 62.2min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140625) |
| base-b-test-4-npu-a3 / run (1) | 62.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140645) |
| base-b-test-16-npu-a3 / run (0) | 62.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140660) |
| base-b-test-4-npu-a3 / run (0) | 62.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140708) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 62.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140967) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 62.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140995) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 62.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093141025) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 62.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093141055) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140582

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源已被删除或路径错误，属于环境配置或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140614

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140625

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140645

- **base-b-test-16-npu-a3 / run (0)**: 作业失败原因是Azure Blob存储返回BlobNotFound错误，表明CI所需的某个文件或工件在存储中缺失或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140660

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140708

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140967

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的依赖文件或缓存已缺失或路径错误，可能是资源被清理或配置有误，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140995

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093141025

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093141055

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 39.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140540) |
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31590137211/job/94093140564) |


## [Run #31589699557](https://github.com/sgl-project/sglang/actions/runs/31589699557)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 6.4min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31589699557

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729295) |
| base-b-test-1-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729304) |
| base-b-test-4-npu-a3 / run (1) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729348) |
| base-b-test-4-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729356) |
| base-b-test-2-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729368) |
| base-b-test-16-npu-a3 / run (0) | 5.3min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729395) |
| base-b-test-8-npu-a3 / run (0) | 5.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729401) |
| base-a-test-1-npu-a2 / run (0) | 5.0min | 环境问题 | 下载triton-ascend依赖时网络连接中断，导致容器执行失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729463) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729711) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729728) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729768) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 5.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729783) |

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions环境初始化、Node.js弃用警告及上传diffusion-failures工件时未找到文件，未包含多模态生成测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729295

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729304

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729348

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是上游作业未成功上传或存储过期，需检查相关依赖资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729356

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729368

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问一个 Azure Blob 中的日志文件，但该 blob 不存在（BlobNotFound）。这通常是日志上传延迟、文件被清理或路径配置错误所致，属于基础设施或环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729395

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729401

- **base-a-test-1-npu-a2 / run (0)**: 在安装triton-ascend==3.2.1.dev20260530时，下载188.5MB的whl包过程中连接中断，重试后仍失败，最终触发自定义容器执行错误，作业终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729463

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是上传失败、过期或被误删，需检查存储配置及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729711

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729728

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729768

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31589699557/job/94091729783


## [Run #31588931683](https://github.com/sgl-project/sglang/actions/runs/31588931683)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 10.9min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31588931683

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.6min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289853) |
| base-b-test-16-npu-a3 / run (0) | 9.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289908) |
| base-b-test-1-npu-a3 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289973) |
| base-b-test-8-npu-a3 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289977) |
| base-b-test-2-npu-a3 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289979) |
| base-b-test-4-npu-a3 / run (0) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289988) |
| base-b-test-4-npu-a3 / run (1) | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290001) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290194) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290222) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290267) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 9.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290282) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败产物，但作业最终失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289853

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储账户中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施或配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289908

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或存储配置变更，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289973

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是上游产物未上传或过期，需检查相关依赖资源是否有效。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289977

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289979

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289988

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290001

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在 Azure Blob 中缺失或路径错误，可能是资源未上传或已被删除，需检查存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290194

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传、被清理或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290222

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重、测试数据或构建产物）在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290267

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089290282

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588931683/job/94089289967) |


## [Run #31588342274](https://github.com/sgl-project/sglang/actions/runs/31588342274)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 8.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31588342274

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087468742) |
| base-b-test-2-npu-a3 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087468920) |
| base-b-test-8-npu-a3 / run (0) | 7.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087468960) |
| base-b-test-1-npu-a3 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087468965) |
| base-b-test-16-npu-a3 / run (0) | 7.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469013) |
| base-a-test-1-npu-a2 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469064) |
| base-b-test-4-npu-a3 / run (1) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469067) |
| base-b-test-4-npu-a3 / run (0) | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469123) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469910) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469949) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087470051) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087470064) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087468742

- **base-b-test-2-npu-a3 / run (0)**: 作业运行时尝试下载日志文件，但 Azure Blob 返回 BlobNotFound 错误，说明日志文件已被删除或路径错误，属于基础设施或配置问题，与代码本身无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087468920

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中已被删除或路径错误，属于基础设施或配置问题，需检查上传步骤或存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087468960

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087468965

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或依赖文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469013

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469064

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469067

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469123

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469910

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087469949

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087470051

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件（如模型权重或缓存）在存储中缺失或路径错误，可能是资源未上传或已被清理，需检查相关配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588342274/job/94087470064


## [Run #31588237155](https://github.com/sgl-project/sglang/actions/runs/31588237155)
- **分支**: `codex/cosmos3-qknorm-rope-fusion-pr`
- **总耗时**: 281.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31588237155

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 115.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常导致作业中断。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034585) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 22.1min | 性能回归 | NPU性能测试未达预期，minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94128788365) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 作业因健康检查发现其他根因作业失败而被快速跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94136575508) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94140815840) |

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但在15:18:37时出现"Executing the custom container implementation failed"错误，提示联系自托管runner管理员，属于runner环境问题而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034585

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行1128秒后失败，退出码1。该测试为性能测试，失败原因可能是性能未达到50ms的延迟目标，属于性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94128788365

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示健康检查检测到同运行中的base-c-test-perf-8-npu-a3作业失败，触发fast-fail机制，导致本作业未实际执行即被终止，属于依赖的上游作业失败引发的连锁跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94136575508

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 该作业在启动阶段被健康检查快速失败机制跳过，原因是同PR中base-c-test-perf-8-npu-a3作业已失败，本作业被判定为级联失败，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94140815840

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 25.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034347) |
| base-a-test-1-npu-a2 / run (0) | 6.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034364) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034388) |
| base-b-test-8-npu-a3 / run (0) | 12.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034395) |
| base-b-test-4-npu-a3 / run (0) | 29.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034410) |
| base-b-test-2-npu-a3 / run (0) | 19.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034417) |
| base-b-test-4-npu-a3 / run (1) | 14.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034432) |
| base-b-test-16-npu-a3 / run (0) | 50.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034492) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034560) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034575) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31588237155/job/94087034649) |


## [Run #31587676877](https://github.com/sgl-project/sglang/actions/runs/31587676877)
- **分支**: `codex/k3-mm-cache-lease`
- **总耗时**: 9.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31587676877

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.4min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345316) |
| base-b-test-1-npu-a3 / run (0) | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345380) |
| base-b-test-16-npu-a3 / run (0) | 8.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345392) |
| base-b-test-2-npu-a3 / run (0) | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345446) |
| base-a-test-1-npu-a2 / run (0) | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345533) |
| base-b-test-8-npu-a3 / run (0) | 8.4min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345539) |
| base-b-test-4-npu-a3 / run (0) | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345716) |
| base-b-test-4-npu-a3 / run (1) | 8.4min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345743) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085346352) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 8.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085346618) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 8.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085346691) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 8.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085346830) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能确认上传diffusion-failures目录时未找到文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志定位具体错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345316

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是上游构建未成功上传或存储配置变更所致。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345380

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345392

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345446

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345533

- **base-b-test-8-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误，或相关文件已被删除。这属于基础设施/环境配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345539

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，属于环境或资源缺失问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345716

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 blob 资源已被删除或路径错误，属于外部依赖缺失的环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085345743

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085346352

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085346618

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085346691

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/31587676877/job/94085346830


## [Run #31586629612](https://github.com/sgl-project/sglang/actions/runs/31586629612)
- **分支**: `fix/rope-config-and-vl-weight-loading`
- **总耗时**: 12.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31586629612

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 3.2min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941122) |
| base-b-test-1-npu-a3 / run (0) | 12.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941125) |
| base-b-test-8-npu-a3 / run (0) | 12.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941173) |
| base-b-test-2-npu-a3 / run (0) | 12.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941190) |
| base-b-test-16-npu-a3 / run (0) | 12.0min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941198) |
| base-a-test-1-npu-a2 / run (0) | 12.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941199) |
| base-b-test-4-npu-a3 / run (0) | 12.0min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941210) |
| base-b-test-4-npu-a3 / run (1) | 12.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941287) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941474) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941507) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941522) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941570) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅显示上传diffusion-failures目录时提示无文件，说明测试可能未产生失败产物，但作业最终失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941122

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941125

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的工件/缓存文件在 Azure Blob 中不存在，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941173

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/日志文件在存储中缺失，可能是上游任务未成功上传或路径配置错误，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941190

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载或访问的Azure Blob存储对象已被删除或路径错误，可能是构建产物或依赖文件缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941198

- **base-a-test-1-npu-a2 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源过期或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941199

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 系统尝试下载的日志 blob 已被删除或路径错误，属于基础设施/存储配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941210

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941287

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理、上传失败或配置错误，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941474

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941507

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941522

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或测试数据）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586629612/job/94081941570


## [Run #31586416120](https://github.com/sgl-project/sglang/actions/runs/31586416120)
- **分支**: `feature/load-reporter`
- **总耗时**: 200.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31586416120

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 0.6min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314727) |
| base-b-test-4-npu-a3 / run (0) | 1.0min | 其他 | 健康检查中的lint检查失败导致作业快速终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314744) |
| base-b-test-4-npu-a3 / run (1) | 1.0min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314756) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314784) |
| base-b-test-2-npu-a3 / run (0) | 0.9min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314855) |
| base-a-test-1-npu-a2 / run (0) | 0.7min | 其他 | 健康检查失败：lint检查未通过导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314880) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314966) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.9min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081315022) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081315023) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查失败：lint检查未通过导致快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081315038) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.5min | 其他 | 健康检查前置步骤失败，lint检查未通过导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081315059) |

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314727

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动阶段执行health-check时，lint检查结论为failure，触发了fast-fail机制，作业提前结束，未进入实际NPU测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314744

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314756

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业立即终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314784

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314855

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314880

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314966

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查结论为failure，触发fast-fail机制，作业未进入实际测试阶段即退出。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081315022

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行健康检查时，检测到lint检查结论为failure，触发fast-fail机制，作业立即终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081315023

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行PR健康检查时，检测到lint检查结论为failure，触发了fast-fail机制，作业在运行测试前即被终止，并非测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081315038

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查状态为failure，触发了fast-fail机制，作业在真正运行测试前即终止，属于前置检查拦截，非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081315059

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586416120/job/94081314638) |


## [Run #31586352481](https://github.com/sgl-project/sglang/actions/runs/31586352481)
- **分支**: `dev`
- **总耗时**: 281.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31586352481

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.8min | 性能回归 | NPU性能测试未达标导致失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94125050333) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 其他 | 健康检查发现同PR中另一个作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94132793167) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.9min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94136029740) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 0.9min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94155837536) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py执行约1086秒后失败，0/1通过，属于性能测试未达到预期标准（如吞吐或延迟要求），非环境或代码错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94125050333

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 日志显示health-check检测到base-c-test-perf-8-npu-a3作业失败，将其标记为根因失败，并触发fast-fail跳过当前作业。本作业本身未执行任何测试，属于被关联作业失败导致的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94132793167

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示健康检查检测到 base-c-test-perf-8-npu-a3 作业失败，本作业作为级联失败被过滤，但最终因根因作业失败而快速失败，未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94136029740

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 日志显示本作业在启动阶段即被健康检查快速失败机制终止，原因是同批次中base-c-test-perf-8-npu-a3作业失败被判定为根因，本作业作为级联失败被跳过，并非自身执行问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94155837536

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081084846) |
| base-b-test-2-npu-a3 / run (0) | 20.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081084906) |
| base-b-test-4-npu-a3 / run (0) | 32.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081084924) |
| base-b-test-16-npu-a3 / run (0) | 55.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081084937) |
| base-b-test-1-npu-a3 / run (0) | 23.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081084970) |
| base-b-test-8-npu-a3 / run (0) | 8.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081084999) |
| base-b-test-4-npu-a3 / run (1) | 15.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081085006) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 8.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081085292) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 23.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081085295) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 106.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081085356) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 42.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31586352481/job/94081085465) |


## [Run #31585698126](https://github.com/sgl-project/sglang/actions/runs/31585698126)
- **分支**: `agent-cutedsl-moe-nvfp4-sglang`
- **总耗时**: 263.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/31585698126

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 21.9min | 性能回归 | NPU性能测试未达预期，minimax_m2_5模型测试失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94122245212) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1.1min | 环境问题 | 健康检查发现其他作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94129370928) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业（8-npu）已失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94133722379) |
| base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3 | 1.0min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94149109648) |

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 测试test_npu_minimax_m2_5_w8a8_4p_in64k_out1k_prefix90_50ms.py运行1074秒后退出码为1，该测试为性能测试，失败表明性能指标未达标，属于性能回归问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94122245212

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 本作业因健康检查检测到同批次base-c-test-perf-8-npu-a3作业失败，被快速失败机制跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94129370928

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示本作业在启动阶段即被健康检查拦截，原因是同PR中base-c-test-perf-8-npu-a3作业已失败，触发fast-fail机制，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94133722379

- **base-c-test-perf-2-npu-a3 / base-c-test-perf-2-npu-a3**: 该作业在健康检查阶段检测到根因作业base-c-test-perf-8-npu-a3失败，触发快速失败机制，本作业被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94149109648

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| multimodal-gen-test-1-npu-a3 | 22.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080306411) |
| base-b-test-2-npu-a3 / run (0) | 18.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080306580) |
| base-b-test-4-npu-a3 / run (1) | 14.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080306626) |
| base-b-test-16-npu-a3 / run (0) | 44.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080306639) |
| base-b-test-8-npu-a3 / run (0) | 6.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080306640) |
| base-b-test-1-npu-a3 / run (0) | 25.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080306645) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080306742) |
| base-b-test-4-npu-a3 / run (0) | 31.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080306811) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 85.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080307049) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080307143) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 41.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080307167) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 24.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/31585698126/job/94080307359) |


---
*Auto-generated by npu_pr_monitor.py*