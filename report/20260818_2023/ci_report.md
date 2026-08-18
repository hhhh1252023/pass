# NPU CI 执行监控
**生成时间**: 2026-08-18 12:23 UTC
**分析 Run 数**: 28

---

## 📊 本次执行总结

- **成功 Job 数**: 26
- **失败 Run 数**: 28
- **成功 Job 平均耗时**: 11.6min

### ✅ 耗时最长的成功 Job（Top 10）

| Job 名称 | 耗时 | 所属 Run | 链接 |
|----------|------|----------|------|
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 49.0min | #32108944959 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147452) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 33.5min | #32108944959 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95632625168) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 30.3min | #32108944959 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147773) |
| base-b-test-4-npu-a3 / run (0) | 29.8min | #32108944959 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147302) |
| base-b-test-1-npu-a3 / run (0) | 23.4min | #32108944959 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147019) |
| base-b-test-2-npu-a3 / run (0) | 16.3min | #32108944959 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147114) |
| base-b-test-4-npu-a3 / run (1) | 12.7min | #32108944959 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147299) |
| base-a-test-1-npu-a2 / run (0) | 7.6min | #32108483104 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790506) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | #32108944959 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147148) |
| base-a-test-1-npu-a2 / run (0) | 6.4min | #32107630853 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285109) |

---

## 📋 各任务执行统计

| 任务名称 | 执行次数 | 成功 | 执行失败 | 健康检查失败 | 取消 |
|----------|----------|------|---------|-------------|------|
| multimodal-gen-test-1-npu-a3 | 28 | 0 | 21 | 0 | 7 |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 22 | 0 | 0 | 14 | 8 |
| base-b-test-16-npu-a3 / run (0) | 22 | 0 | 0 | 13 | 9 |
| base-b-test-1-npu-a3 / run (0) | 22 | 1 | 0 | 13 | 8 |
| base-b-test-8-npu-a3 / run (0) | 22 | 1 | 0 | 13 | 8 |
| base-b-test-4-npu-a3 / run (0) | 22 | 1 | 0 | 13 | 8 |
| base-b-test-2-npu-a3 / run (0) | 22 | 1 | 0 | 13 | 8 |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 22 | 1 | 0 | 13 | 8 |
| base-b-test-4-npu-a3 / run (1) | 22 | 1 | 0 | 12 | 9 |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 22 | 2 | 0 | 12 | 8 |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 22 | 1 | 0 | 12 | 9 |
| base-a-test-1-npu-a2 / run (0) | 22 | 16 | 0 | 6 | 0 |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 1 | 0 | 0 | 1 | 0 |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 1 | 0 | 0 | 1 | 0 |

---

## 📋 各用例失败统计

*本次分析未发现失败用例。*

---

### ⚠️ 失败的 CI Run

| Run ID | 分支 | 耗时 | 失败任务数 | 失败的任务 | 结论 | 链接 |
|--------|------|------|-----------|-----------|------|------|
| #32108944959<br>[#35222 [CPU] Enable ERNIE models on CPU](https://github.com/sgl-project/sglang/pull/35222) | `ernie-enable` | 120.0min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32108944959) |
| #32106843246<br>[#35071 [PD] Overlap prefill DP-rank bootstrap queries](https://github.com/sgl-project/sglang/pull/35071) | `agentx-upstream/query-prefill-overlap-20260816` | 68.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32106843246) |
| #32106333853<br>[#33684 [Weight Cache] Support static DP/EP layouts](https://github.com/sgl-project/sglang/pull/33684) | `unidy2002/weight-cache-static-dp-ep` | 59.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32106333853) |
| #32110153931<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `codex/diffusion-reuse-srt-qwen-vision` | 59.2min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32110153931) |
| #32108341066<br>[#35228 [Quant] Load compressed-tensors quantized lm_head instead of value-casting it](https://github.com/sgl-project/sglang/pull/35228) | `fix/quantized-lm-head-load` | 57.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32108341066) |
| #32110424589<br>[#34890 [Perf] Hoist DSv4 draft-extend SWA write locs; unify SWA graph buffer naming](https://github.com/sgl-project/sglang/pull/34890) | `main` | 51.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32110424589) |
| #32108108559<br>[#34197 [diffusion] RL rollout support for the Cosmos3 pipeline](https://github.com/sgl-project/sglang/pull/34197) | `feat/cosmos3-rl-rollout-v2` | 47.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32108108559) |
| #32112813910<br>[#34690 [BugFix][VLM] keep Qwen3-VL MoE inference deepstack order](https://github.com/sgl-project/sglang/pull/34690) | `py/fix-qwen3-grounding-acc` | 46.3min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32112813910) |
| #32112853445<br>[#30909 [HiSparse] Fix inflated full token usage for DeepSeek V4](https://github.com/sgl-project/sglang/pull/30909) | `fix-dsv4-hisparse-report` | 45.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32112853445) |
| #32107630853<br>[#33057 fix(xpu): enable compressed-tensors FP8 W8A8 on XPU (RedHatAI FP8-dynamic models)](https://github.com/sgl-project/sglang/pull/33057) | `fix/xpu-compressed-tensors-fp8-w8a8` | 45.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32107630853) |
| #32111934476<br>[#31470 [NVIDIA] Support flashinfer Mega Moe](https://github.com/sgl-project/sglang/pull/31470) | `mega_moe_flashinfer` | 44.0min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32111934476) |
| #32113874133<br>[#32984 [MLX] Upgrade to Torch 2.13/MLX 0.32+ and redesign the Torch-MLX tensor bridge](https://github.com/sgl-project/sglang/pull/32984) | `mlx-032-torch-213-bridge` | 42.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32113874133) |
| #32113475616<br>[#31218 [DSA] Page-aware move_kv_cache for the DSA indexer cache (fix page_size>1 corruption)](https://github.com/sgl-project/sglang/pull/31218) | `fix/dsa-movekv-page-aware` | 41.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32113475616) |
| #32113831439<br>[#35125 [Rust Server] Add e2e latency metadata and fix Sarashina import](https://github.com/sgl-project/sglang/pull/35125) | `fix/rust-server-meta-and-sarashina-import` | 41.4min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32113831439) |
| #32110584572<br>[#32984 [MLX] Upgrade to Torch 2.13/MLX 0.32+ and redesign the Torch-MLX tensor bridge](https://github.com/sgl-project/sglang/pull/32984) | `mlx-032-torch-213-bridge` | 41.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32110584572) |
| #32108483104<br>[#35172 [Quantization] Extract shared checkpoint quant metadata resolver](https://github.com/sgl-project/sglang/pull/35172) | `codex/checkpoint-quant-spec` | 39.5min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32108483104) |
| #32109119540<br>[#35238 Exclude multimodal-gen NPU jobs from fast-fail cascade](https://github.com/sgl-project/sglang/pull/35238) | `patch-8` | 34.8min | 5 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32109119540) |
| #32109803258<br>[#33880 [diffusion] optimization: reduce minimax h3 mps memory pressure](https://github.com/sgl-project/sglang/pull/33880) | `codex/minimax-h3-mps` | 30.9min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32109803258) |
| #32107345516<br>[#34993 [diffusion] fix: make MiniMax-H3 AdaLN cache rebuild transactional](https://github.com/sgl-project/sglang/pull/34993) | `main` | 15.7min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32107345516) |
| #32106864860<br>[#34581 [Diffusion] Optimizing MiniMax-H3 for consumer-level GPUs: INT8 Linear + pluggable DiT attention backends](https://github.com/sgl-project/sglang/pull/34581) | `experiment/h3-single-gpu-offload` | 15.6min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32106864860) |
| #32112654642<br>[#35125 [Rust Server] Add e2e latency metadata and fix Sarashina import](https://github.com/sgl-project/sglang/pull/35125) | `fix/rust-server-meta-and-sarashina-import` | 14.8min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-16-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32112654642) |
| #32109008510<br>[#34498 [ROCm] Direct-write a8w8 bmm output to eliminate o_proj transpose copy](https://github.com/sgl-project/sglang/pull/34498) | `opt/kimi-k2-mxfp4-fp8-bmm-direct-write` | 14.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-1-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32109008510) |
| #32107054597<br>[#35050 [XPU] Fix decode graph runner is_current_stream_capturing on non-CUDA devices](https://github.com/sgl-project/sglang/pull/35050) | `main` | 13.0min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-4-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-1-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32107054597) |
| #32109243958<br>[#35006 [Diffusion] Reuse SRT Qwen vision and text modules](https://github.com/sgl-project/sglang/pull/35006) | `codex/diffusion-reuse-srt-qwen-vision` | 12.3min | 11 | multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-b-test-8-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32109243958) |
| #32107957447<br>[#34933 [diffusion] Per-section LoRA adapters on fused linear layers](https://github.com/sgl-project/sglang/pull/34933) | `main` | 12.1min | 1 | multimodal-gen-test-1-npu-a3 | failure | [run link](https://github.com/sgl-project/sglang/actions/runs/32107957447) |
| #32109660775<br>[#35214 [DSV4] Turn on mhc post pre fusion by default](https://github.com/sgl-project/sglang/pull/35214) | `main` | 8.6min | 11 | base-b-test-1-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-8-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (0), base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3, base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32109660775) |
| #32105615176<br>[#35225 refactor: rename chat response token IDs](https://github.com/sgl-project/sglang/pull/35225) | `main` | 7.2min | 11 | base-b-test-8-npu-a3 / run (0), base-b-test-16-npu-a3 / run (0), base-b-test-2-npu-a3 / run (0), base-b-test-4-npu-a3 / run (1), base-b-test-4-npu-a3 / run (0), multimodal-gen-test-1-npu-a3, base-b-test-1-npu-a3 / run (0), base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3, base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3, base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3, base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32105615176) |
| #32106097016<br>[#31453 [Diffusion][Refactor] Refactor and extract complex RoPE implementation to layers/rotary_embedding for MOVA DiT](https://github.com/sgl-project/sglang/pull/31453) | `main` | 5.2min | 1 | multimodal-gen-test-1-npu-a3 | cancelled | [run link](https://github.com/sgl-project/sglang/actions/runs/32106097016) |

---


## [Run #32113874133](https://github.com/sgl-project/sglang/actions/runs/32113874133)
- **分支**: `mlx-032-torch-213-bridge`
- **总耗时**: 42.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32113874133

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.7min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅见上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117292) |
| base-b-test-16-npu-a3 / run (0) | 2.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117363) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117442) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117469) |
| base-b-test-4-npu-a3 / run (0) | 2.0min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117488) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117511) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117556) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117837) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117894) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.0min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117935) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117987) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败产物，但作业最终失败原因不明，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117292

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117363

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到同一次运行中的multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117442

- **base-b-test-8-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3为根因失败，本作业因级联失败被过滤，实际未执行测试，属于CI流程的快速失败保护。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117469

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117488

- **base-b-test-2-npu-a3 / run (0)**: 健康检查显示multimodal-gen-test-1-npu-a3为根因失败，本作业因级联失败被过滤并快速失败，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117511

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因作业，导致本作业（base-b-test-4-npu-a3）被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117556

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117837

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117894

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤并快速失败，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117935

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117987

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32113874133/job/95639117481) |


## [Run #32113831439](https://github.com/sgl-project/sglang/actions/runs/32113831439)
- **分支**: `fix/rust-server-meta-and-sarashina-import`
- **总耗时**: 41.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32113831439

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 1.5min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007495) |
| base-a-test-1-npu-a2 / run (0) | 0.7min | 其他 | 健康检查发现其他作业失败，触发快速失败机制 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007497) |
| multimodal-gen-test-1-npu-a3 | 10.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007521) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007570) |
| base-b-test-16-npu-a3 / run (0) | 1.8min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007577) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007597) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007630) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007715) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007945) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | PR健康检查失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007972) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | PR测试健康检查失败，根因是多模态测试任务失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639008143) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639008163) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败被取消，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007495

- **base-a-test-1-npu-a2 / run (0)**: 该作业在启动前进行健康检查，发现同一PR中的multimodal-gen-test-1-npu-a3作业已失败，被判定为根因失败，因此本作业被快速跳过（fast-fail），并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007497

- **multimodal-gen-test-1-npu-a3**: 日志仅显示GitHub Actions运行器初始化、Node版本警告及上传artifact时未找到diffusion-failures目录，未包含multimodal-gen测试的实际执行结果或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007521

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007570

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007577

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007597

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007630

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007715

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007945

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639007972

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败任务，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639008143

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113831439/job/95639008163


## [Run #32113475616](https://github.com/sgl-project/sglang/actions/runs/32113475616)
- **分支**: `fix/dsa-movekv-page-aware`
- **总耗时**: 41.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32113475616

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880523) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880652) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880698) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880755) |
| base-b-test-16-npu-a3 / run (0) | 1.3min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880810) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880834) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880966) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880997) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | PR测试健康检查失败，根因是多模态生成测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637881146) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查快速失败，因同一PR中另一个作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637881162) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 1.0min | 其他 | PR测试健康检查失败，根因是多模态生成测试作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637881241) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示runner启动、Node版本警告及上传artifact时无文件。可能测试未运行或日志被截断，需查看完整日志以确定失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880523

- **base-b-test-1-npu-a3 / run (0)**: 该作业在健康检查阶段检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被跳过并报错，属于级联失败，非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880652

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤，最终因快速失败策略被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880698

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880755

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880810

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，将其识别为根因，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880834

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，将其判定为根因，本作业因快速失败机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880966

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880997

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637881146

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止，属于依赖的根因作业失败导致的级联跳过。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637881162

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637881241

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32113475616/job/95637880663) |


## [Run #32112853445](https://github.com/sgl-project/sglang/actions/runs/32112853445)
- **分支**: `fix-dsv4-hisparse-report`
- **总耗时**: 45.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32112853445

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.4min | 其他 | 作业日志被截断，未显示实际测试结果，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998020) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998181) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998275) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998276) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998337) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998355) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998409) |
| base-b-test-16-npu-a3 / run (0) | 4.2min | 环境问题 | GitHub Actions 下载依赖超时导致作业失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998499) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998975) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635999007) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635999024) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.4min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635999223) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后上传diffusion-failures目录时提示无文件，说明测试可能通过或未产生失败样本，但无法确认具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998020

- **base-b-test-8-npu-a3 / run (0)**: 该作业因健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败而跳过，属于级联失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998181

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败（根因），本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998275

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业（base-b-test-1-npu-a3）因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998276

- **base-a-test-1-npu-a2 / run (0)**: 健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-a-test-1-npu-a2）被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998337

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998355

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998409

- **base-b-test-16-npu-a3 / run (0)**: 下载 actions/github-script@v8 时因 HttpClient.Timeout 100秒超时失败，重试后仍失败，最终导致健康检查脚本无法执行，作业以非零退出码结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998499

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业在启动前的PR健康检查中发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业被跳过未实际执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635998975

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635999007

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635999024

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业在“Check PR test health”步骤被跳过，因根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112853445/job/95635999223


## [Run #32112813910](https://github.com/sgl-project/sglang/actions/runs/32112813910)
- **分支**: `py/fix-qwen3-grounding-acc`
- **总耗时**: 46.3min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32112813910

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 11.9min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164183) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164344) |
| base-a-test-1-npu-a2 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164461) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因任务失败，导致级联跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164477) |
| base-b-test-16-npu-a3 / run (0) | 3.2min | 其他 | 健康检查发现根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164486) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164487) |
| base-b-test-1-npu-a3 / run (0) | 0.6min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164515) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现根因任务失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164566) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164911) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 3.7min | 其他 | PR测试健康检查失败，根因是多模态生成测试任务失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164930) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | PR健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164983) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637165005) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行的具体输出或错误信息，仅显示上传diffusion-failures工件时未找到文件，说明测试可能未产生失败样本，但无法确定具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164183

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164344

- **base-a-test-1-npu-a2 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业（base-a-test-1-npu-a2）被快速失败机制跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164461

- **base-b-test-4-npu-a3 / run (0)**: 日志显示本作业因健康检查过滤级联失败而被跳过，根因是multimodal-gen-test-1-npu-a3任务失败，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164477

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业作为依赖被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164486

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被终止，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164487

- **base-b-test-1-npu-a3 / run (0)**: 作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，按策略快速失败，未执行实际测试，属于级联跳过而非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164515

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败任务，最终根因失败任务为multimodal-gen-test-1-npu-a3，本作业因快速失败被取消，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164566

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动前的PR测试健康检查中，检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业未实际运行即被跳过，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164911

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败任务，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164930

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637164983

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112813910/job/95637165005


## [Run #32112654642](https://github.com/sgl-project/sglang/actions/runs/32112654642)
- **分支**: `fix/rust-server-meta-and-sarashina-import`
- **总耗时**: 14.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32112654642

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 6.4min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359660) |
| base-a-test-1-npu-a2 / run (0) | 5.8min | 环境问题 | 自定义容器执行失败，NPU测试未运行 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359691) |
| base-b-test-16-npu-a3 / run (0) | 14.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359724) |
| base-b-test-8-npu-a3 / run (0) | 14.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359772) |
| base-b-test-2-npu-a3 / run (0) | 14.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359774) |
| base-b-test-1-npu-a3 / run (0) | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359889) |
| base-b-test-4-npu-a3 / run (1) | 14.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359900) |
| base-b-test-4-npu-a3 / run (0) | 14.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359964) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 14.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635360049) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635360257) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635360279) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 14.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635360306) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node版本弃用警告及上传artifact步骤，未出现测试执行或失败断言，无法判断具体失败原因，可能为日志截断或作业被外部终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359660

- **base-a-test-1-npu-a2 / run (0)**: 作业在启动测试前，执行自定义容器实现时失败，错误提示联系自托管runner管理员，属于基础设施/容器环境问题，非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359691

- **base-b-test-16-npu-a3 / run (0)**: 作业失败原因是Azure Blob存储返回BlobNotFound错误，即请求的资源不存在。这通常是由于CI配置中引用的文件或工件未正确上传、路径错误或已被删除，属于环境或基础设施问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359724

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359772

- **base-b-test-2-npu-a3 / run (0)**: 作业在下载或访问某个Azure Blob存储中的文件时，返回BlobNotFound错误，说明该文件已被删除或路径错误，属于外部依赖资源缺失的环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359774

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 资源缺失，可能是构建产物未上传或路径错误，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359889

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359900

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635359964

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635360049

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635360257

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或资源在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635360279

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理或路径错误，需检查存储配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32112654642/job/95635360306


## [Run #32111934476](https://github.com/sgl-project/sglang/actions/runs/32111934476)
- **分支**: `mega_moe_flashinfer`
- **总耗时**: 44.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32111934476

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190785) |
| base-b-test-2-npu-a3 / run (0) | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190877) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190891) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190923) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190990) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他作业根因失败而跳过本作业 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191043) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191101) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.6min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191375) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败被快速失败跳过，并非自身测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191588) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191600) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191637) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传diffusion-failures目录时未找到文件，可能测试未运行或提前退出，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190785

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，根因过滤后仍存在该失败作业，因此本作业被快速失败跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190877

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业为multimodal-gen-test-1-npu-a3，本作业（base-b-test-1-npu-a3）因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190891

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业（base-b-test-4-npu-a3）被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190923

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190990

- **base-b-test-8-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制主动跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191043

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191101

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 该作业在启动前的PR健康检查中检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191375

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因作业为multimodal-gen-test-1-npu-a3，本作业被Fast-fail机制跳过，未执行实际测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191588

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业在PR测试健康检查阶段被跳过，根因是multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191600

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示本作业在启动前被健康检查过滤，根因是multimodal-gen-test-1-npu-a3作业失败，导致本作业被Fast-fail跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633191637

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32111934476/job/95633190989) |


## [Run #32110584572](https://github.com/sgl-project/sglang/actions/runs/32110584572)
- **分支**: `mlx-032-torch-213-bridge`
- **总耗时**: 41.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32110584572

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 5.8min | 环境问题 | GitHub Actions 下载 upload-artifact 动作超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991499) |
| base-b-test-4-npu-a3 / run (1) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991804) |
| base-b-test-2-npu-a3 / run (0) | 40.8min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991835) |
| base-b-test-1-npu-a3 / run (0) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991837) |
| base-b-test-4-npu-a3 / run (0) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991872) |
| base-b-test-8-npu-a3 / run (0) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991956) |
| base-b-test-16-npu-a3 / run (0) | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991993) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628992390) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628992426) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628992476) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 40.8min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628992480) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，虽然后续重试成功，但可能影响了作业执行，最终未找到 diffusion-failures 文件，作业提前结束。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991499

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是缓存或依赖资源缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991804

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI依赖的远程存储对象缺失或路径错误，可能是上传失败、清理策略或配置变更所致，需检查相关存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991835

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的工件/文件在 Azure Blob 存储中已被删除或路径错误，属于基础设施或配置问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991837

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或依赖文件在 Azure Blob 存储中缺失，可能是文件被清理、路径错误或上传失败，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991872

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，可能是 CI 脚本尝试下载或访问的工件/缓存文件已被删除或路径错误，属于外部存储依赖问题，非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991956

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关配置或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991993

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是上传失败、资源被删除或配置错误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628992390

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628992426

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及文件是否存在。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628992476

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源未上传、被删除或配置有误，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628992480

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32110584572/job/95628991866) |


## [Run #32110424589](https://github.com/sgl-project/sglang/actions/runs/32110424589)
- **分支**: `main`
- **总耗时**: 51.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32110424589

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 34.9min | 其他 | 作业日志不完整，未显示测试执行过程，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576044) |
| base-b-test-16-npu-a3 / run (0) | 51.0min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576130) |
| base-b-test-8-npu-a3 / run (0) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576154) |
| base-b-test-4-npu-a3 / run (0) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576163) |
| base-b-test-2-npu-a3 / run (0) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576218) |
| base-b-test-4-npu-a3 / run (1) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576237) |
| base-b-test-1-npu-a3 / run (0) | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576310) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576648) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576649) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576702) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 51.0min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576704) |

- **multimodal-gen-test-1-npu-a3**: 日志中只有GitHub Actions的初始化、Node版本警告和上传artifact步骤，未包含实际测试命令或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576044

- **base-b-test-16-npu-a3 / run (0)**: 作业日志显示BlobNotFound错误，说明CI流程尝试下载的blob（可能为测试数据或构建产物）已被删除或路径错误，属于基础设施/环境配置问题，需检查相关存储路径或重跑作业。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576130

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576154

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576163

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576218

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576237

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576310

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的 Azure Blob 存储中的某个文件缺失或路径错误，可能是资源未上传、被删除或配置有误，需检查相关存储路径和上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576648

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576649

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576702

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储文件缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576704

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.1min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32110424589/job/95628576138) |


## [Run #32110153931](https://github.com/sgl-project/sglang/actions/runs/32110153931)
- **分支**: `codex/diffusion-reuse-srt-qwen-vision`
- **总耗时**: 59.2min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32110153931

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.9min | 其他 | 日志被截断，未显示实际测试失败原因，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856400) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856532) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856552) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856633) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查发现根因任务失败，导致级联跳过本作业。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856680) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因是其他作业失败导致级联取消 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856712) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856757) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856969) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627857005) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 环境问题 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627857035) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.0min | 其他 | 健康检查失败，根因作业为multimodal-gen-test-1-npu-a3，本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627857047) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法定位具体失败点。作业最终上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业仍标记失败，需查看完整日志确认。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856400

- **base-b-test-8-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制，本作业被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856532

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856552

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到同一PR中的multimodal-gen-test-1-npu-a3作业失败，根据快速失败策略，本作业被跳过并报错退出，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856633

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败任务，根因是多模态生成测试（multimodal-gen-test-1-npu-a3）失败，本作业被快速失败机制跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856680

- **base-b-test-4-npu-a3 / run (0)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856712

- **base-b-test-4-npu-a3 / run (1)**: 本作业在健康检查阶段检测到根因作业multimodal-gen-test-1-npu-a3失败，按策略快速失败，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856757

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查过滤了多个级联失败，根因是multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856969

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627857005

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动前的PR健康检查中检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被终止，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627857035

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败机制跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627857047

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32110153931/job/95627856556) |


## [Run #32109803258](https://github.com/sgl-project/sglang/actions/runs/32109803258)
- **分支**: `codex/minimax-h3-mps`
- **总耗时**: 30.9min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32109803258

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 29.9min | 环境问题 | 作业因缺少失败产物文件而提前结束，未显示实际测试失败原因。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109803258/job/95627165694) |

- **multimodal-gen-test-1-npu-a3**: 日志显示上传diffusion-failures目录时提示无文件，作业在测试阶段后直接清理，未捕获到具体错误。可能因测试未生成失败产物或环境配置问题导致作业异常终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109803258/job/95627165694


## [Run #32109660775](https://github.com/sgl-project/sglang/actions/runs/32109660775)
- **分支**: `main`
- **总耗时**: 8.6min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32109660775

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-1-npu-a3 / run (0) | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626245958) |
| multimodal-gen-test-1-npu-a3 | 7.3min | 环境问题 | GitHub Actions 下载 actions/checkout 时网络超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626245968) |
| base-b-test-8-npu-a3 / run (0) | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626245987) |
| base-b-test-4-npu-a3 / run (1) | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626245998) |
| base-b-test-16-npu-a3 / run (0) | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246026) |
| base-b-test-2-npu-a3 / run (0) | 7.2min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246047) |
| base-a-test-1-npu-a2 / run (0) | 7.0min | 环境问题 | 自定义容器执行失败，导致作业提前终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246065) |
| base-b-test-4-npu-a3 / run (0) | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246071) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246264) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246283) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246291) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 7.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246418) |

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试访问的 Azure Blob 存储资源缺失或路径错误，可能是缓存、依赖或上传文件未正确生成，属于环境配置或资源问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626245958

- **multimodal-gen-test-1-npu-a3**: 日志显示在准备阶段下载 actions/checkout@v4 时，HTTP 请求超时（100秒），重试后仍失败，最终作业无法正常执行。属于网络或 GitHub 服务端临时问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626245968

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626245987

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626245998

- **base-b-test-16-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的存储对象缺失，可能是文件被清理、路径错误或上传失败，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246026

- **base-b-test-2-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是资源被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246047

- **base-a-test-1-npu-a2 / run (0)**: 日志显示在构建sglang包时，自定义容器实现执行失败，错误信息为'Executing the custom container implementation failed'，可能是容器环境或配置问题，而非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246065

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源被清理、上传失败或配置指向了不存在的文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246071

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，属于环境配置或资源管理问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246264

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246283

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246291

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109660775/job/95626246418


## [Run #32109243958](https://github.com/sgl-project/sglang/actions/runs/32109243958)
- **分支**: `codex/diffusion-reuse-srt-qwen-vision`
- **总耗时**: 12.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32109243958

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 9.5min | 其他 | 作业未实际运行测试，仅上传空产物后正常结束。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097438) |
| base-b-test-1-npu-a3 / run (0) | 11.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097558) |
| base-b-test-8-npu-a3 / run (0) | 11.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097560) |
| base-b-test-2-npu-a3 / run (0) | 11.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097566) |
| base-b-test-16-npu-a3 / run (0) | 11.1min | 环境问题 | Azure Blob 存储中指定的日志文件不存在，导致作业无法获取日志而失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097586) |
| base-b-test-4-npu-a3 / run (1) | 11.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097651) |
| base-b-test-4-npu-a3 / run (0) | 11.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097807) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 11.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097852) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 11.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097869) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 11.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097873) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 11.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097926) |

- **multimodal-gen-test-1-npu-a3**: 日志显示作业在准备阶段下载actions后，直接执行上传artifact步骤，未发现任何测试执行或失败信息，最终因无文件上传而正常结束，可能为作业配置或触发条件问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097438

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097558

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是上传失败、路径错误或资源被清理，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097560

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查 blob 名称和存储账户配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097566

- **base-b-test-16-npu-a3 / run (0)**: 作业运行时尝试下载或访问的 blob 资源（可能为日志或依赖文件）在存储中不存在，返回 BlobNotFound 错误。这通常由日志清理、路径错误或上传失败引起，属于基础设施/环境问题，而非代码或性能问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097586

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 文件缺失，可能是构建产物未上传或路径错误，属于基础设施/环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097651

- **base-b-test-4-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097807

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097852

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097869

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097873

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程存储对象缺失或路径错误，可能是资源被清理、上传失败或配置变更，需检查存储路径及上传流程。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097926

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32109243958/job/95625097701) |


## [Run #32109119540](https://github.com/sgl-project/sglang/actions/runs/32109119540)
- **分支**: `patch-8`
- **总耗时**: 34.8min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32109119540

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.1min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490675) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490766) |
| base-b-test-4-npu-a3 / run (1) | 33.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490885) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490888) |
| base-a-test-1-npu-a2 / run (0) | 2.2min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490938) |
| base-b-test-16-npu-a3 / run (0) | 33.9min | 环境问题 | 日志中引用的Azure Blob存储对象不存在，导致作业无法获取必要文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490960) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491005) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491016) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 5.4min | 环境问题 | 自定义容器执行失败，NPU环境初始化异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491543) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 33.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491582) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.0min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491811) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 1.2min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95649145801) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，仅显示上传工件时未找到diffusion-failures目录，可能测试未运行或未产生失败文件，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490675

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了级联失败，根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490766

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 存储文件已被删除或路径错误，可能是缓存或依赖文件缺失，需检查相关存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490885

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动前的健康检查中检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490888

- **base-a-test-1-npu-a2 / run (0)**: 健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490938

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的blob文件已被删除或路径错误，可能是构建产物未上传或存储配置变更，需检查相关上传步骤或存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640490960

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行即被取消，属于级联失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491005

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491016

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在加载模型分片后，TP进程获取环境变量时自定义容器实现失败，提示联系自托管runner管理员，属于NPU容器环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491543

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491582

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示模型权重加载到92%时，自定义容器实现执行失败，提示联系自托管runner管理员，属于runner环境或容器配置问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491811

- **base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95649145801

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 4.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32109119540/job/95640491838) |


## [Run #32109008510](https://github.com/sgl-project/sglang/actions/runs/32109008510)
- **分支**: `opt/kimi-k2-mxfp4-fp8-bmm-direct-write`
- **总耗时**: 14.3min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32109008510

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.8min | 其他 | 作业日志被截断，未显示实际测试失败原因，仅显示上传失败产物时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346243) |
| base-b-test-8-npu-a3 / run (0) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346382) |
| base-b-test-4-npu-a3 / run (1) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346390) |
| base-b-test-1-npu-a3 / run (0) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346392) |
| base-b-test-16-npu-a3 / run (0) | 12.9min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346403) |
| base-b-test-4-npu-a3 / run (0) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346419) |
| base-b-test-2-npu-a3 / run (0) | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346509) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346638) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346659) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346673) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.9min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346702) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行细节。最后仅显示上传diffusion-failures目录时未找到文件，说明测试可能未产生失败产物，但具体失败原因需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346243

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被清理或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346382

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储资源缺失，可能是文件被删除、路径错误或存储配置问题，属于环境依赖问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346390

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载或访问的 Azure Blob 存储对象缺失，可能是文件被删除、路径错误或上传未完成，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346392

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试访问的Azure Blob存储资源缺失，可能是日志上传或下载路径错误，或资源已被删除。这属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346403

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346419

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346509

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试访问的 Azure Blob 存储对象已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346638

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346659

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346673

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失，可能是资源被清理、路径错误或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346702

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32109008510/job/95624346494) |


## [Run #32108944959](https://github.com/sgl-project/sglang/actions/runs/32108944959)
- **分支**: `ernie-enable`
- **总耗时**: 120.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32108944959

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 118.9min | 其他 | 作业日志被截断，未显示实际测试结果，仅看到上传artifact时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624146777) |
| base-b-test-16-npu-a3 / run (0) | 74.3min | 环境问题 | 自定义容器执行失败，导致作业在准备阶段中止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147074) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 83.9min | 环境问题 | 自定义容器执行失败，自托管runner环境异常 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147585) |
| base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3 | 42.2min | 性能回归 | DeepSeek v4 Flash 性能测试未达预期，测试失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95640882985) |
| base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3 | 22.9min | 环境问题 | 自定义容器执行失败，NPU性能测试中途异常终止。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95644460717) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法判断具体失败原因。仅能看到上传diffusion-failures目录时提示无文件，说明测试可能未产生失败样本，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624146777

- **base-b-test-16-npu-a3 / run (0)**: 日志显示在apt更新后约1小时，出现"Executing the custom container implementation failed"错误，属于自托管runner环境问题，非代码或测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147074

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示测试运行正常，但最后出现"Executing the custom container implementation failed"错误，提示联系runner管理员，属于runner容器环境问题而非代码或测试本身失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147585

- **base-c-test-perf-16-npu-a3 / base-c-test-perf-16-npu-a3**: 测试套件中 deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms 性能测试返回退出码1，而 qwen3.5 测试通过，表明该模型性能未达标，可能因代码改动导致性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95640882985

- **base-c-test-perf-4-npu-a3 / base-c-test-perf-4-npu-a3**: 日志显示Prefill正常进行，但随后报错“Executing the custom container implementation failed”，属于自托管runner容器环境问题，非代码或性能回归。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95644460717

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-b-test-1-npu-a3 / run (0) | 23.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147019) |
| base-b-test-2-npu-a3 / run (0) | 16.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147114) |
| base-b-test-8-npu-a3 / run (0) | 6.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147148) |
| base-a-test-1-npu-a2 / run (0) | 5.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147251) |
| base-b-test-4-npu-a3 / run (1) | 12.7min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147299) |
| base-b-test-4-npu-a3 / run (0) | 29.8min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147302) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 3.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147441) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 49.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147452) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 30.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95624147773) |
| base-c-test-perf-8-npu-a3 / base-c-test-perf-8-npu-a3 | 33.5min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108944959/job/95632625168) |


## [Run #32108483104](https://github.com/sgl-project/sglang/actions/runs/32108483104)
- **分支**: `codex/checkpoint-quant-spec`
- **总耗时**: 39.5min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32108483104

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 20.1min | 其他 | 作业因网络超时导致action下载失败，但重试后成功，最终无测试失败产物上传。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790403) |
| base-b-test-4-npu-a3 / run (1) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790468) |
| base-b-test-16-npu-a3 / run (0) | 1.0min | 其他 | 健康检查失败：lint检查未通过导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790483) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790523) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790534) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790535) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 环境问题 | 健康检查中lint检查失败导致作业快速终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790620) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | PR健康检查中lint检查失败导致作业快速失败 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790889) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.3min | 其他 | PR健康检查中的lint检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790920) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | PR健康检查失败，lint检查未通过导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790926) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 环境问题 | PR健康检查中lint检查失败导致作业提前终止 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622791044) |

- **multimodal-gen-test-1-npu-a3**: 日志显示actions/checkout下载时首次HTTP请求超时，重试后成功。后续测试运行正常，仅上传artifact时提示无失败文件，属正常情况，无明确失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790403

- **base-b-test-4-npu-a3 / run (1)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际NPU测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790468

- **base-b-test-16-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790483

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint 检查结论为 failure，触发 fast-fail 机制，作业未进入实际测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790523

- **base-b-test-8-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790534

- **base-b-test-4-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790535

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，lint检查结论为failure，触发了fast-fail机制，作业提前终止，未进入实际测试阶段。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790620

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 作业在启动阶段执行health-check时，检测到PR的lint检查结论为failure，触发fast-fail机制，作业未进入实际测试即终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790889

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查状态为failure，触发了fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790920

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查结果为failure，触发fast-fail机制，作业未进入实际测试阶段即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790926

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 作业在启动阶段执行health-check时，检测到lint检查结论为failure，触发fast-fail机制，作业未进入实际测试即被终止。这是CI前置检查失败，非测试本身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622791044

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 7.6min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108483104/job/95622790506) |


## [Run #32108341066](https://github.com/sgl-project/sglang/actions/runs/32108341066)
- **分支**: `fix/quantized-lm-head-load`
- **总耗时**: 57.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32108341066

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 8.6min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628434989) |
| base-b-test-2-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435052) |
| base-b-test-1-npu-a3 / run (0) | 0.7min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435087) |
| base-b-test-4-npu-a3 / run (0) | 0.8min | 其他 | 健康检查快速失败，根因作业为multimodal-gen-test-1-npu-a3 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435132) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435151) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现根因失败作业，触发快速失败机制。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435212) |
| base-b-test-4-npu-a3 / run (1) | 0.9min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435243) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435432) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 0.8min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败导致级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435433) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.7min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435447) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435492) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的错误信息，仅显示runner启动、依赖下载、上传artifact（无文件）及清理步骤，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628434989

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动前的健康检查中检测到根因作业multimodal-gen-test-1-npu-a3失败，根据快速失败策略跳过本作业，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435052

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动前的健康检查中检测到根因失败作业multimodal-gen-test-1-npu-a3，触发fast-fail机制，本作业未实际运行测试即被取消。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435087

- **base-b-test-4-npu-a3 / run (0)**: 该作业因其他作业（multimodal-gen-test-1-npu-a3）失败被级联过滤，属于健康检查快速失败机制触发，非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435132

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3作业失败，本作业因快速失败机制被取消，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435151

- **base-b-test-8-npu-a3 / run (0)**: 该作业因其他作业（multimodal-gen-test-1-npu-a3）失败而被级联取消，并非自身问题。健康检查过滤了多个级联失败后，确定根因作业并执行fast-fail，导致本作业提前终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435212

- **base-b-test-4-npu-a3 / run (1)**: 健康检查显示根因失败作业为multimodal-gen-test-1-npu-a3，本作业因级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435243

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被过滤并快速失败，并非自身测试问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435432

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 该作业因PR健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，触发fast-fail机制被跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435433

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 作业在启动前的健康检查阶段检测到同一次PR运行中的multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业被跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435447

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435492

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.0min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32108341066/job/95628435123) |


## [Run #32108108559](https://github.com/sgl-project/sglang/actions/runs/32108108559)
- **分支**: `feat/cosmos3-rl-rollout-v2`
- **总耗时**: 47.0min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32108108559

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 25.2min | 其他 | 作业日志被截断，未显示实际测试结果，仅看到上传工件时无文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32108108559/job/95640981146) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法判断测试是否失败。仅看到上传diffusion-failures目录时提示无文件，可能测试未产生失败样本或作业提前结束，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32108108559/job/95640981146


## [Run #32107957447](https://github.com/sgl-project/sglang/actions/runs/32107957447)
- **分支**: `main`
- **总耗时**: 12.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32107957447

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.7min | 环境问题 | 作业因缺少diffusion-failures目录而提前结束，未执行实际测试。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107957447/job/95621323404) |

- **multimodal-gen-test-1-npu-a3**: 日志显示upload-artifact步骤未找到diffusion-failures/文件，说明测试未生成失败产物，作业可能因环境或前置步骤问题未正常运行测试，最终无实际失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107957447/job/95621323404


## [Run #32107630853](https://github.com/sgl-project/sglang/actions/runs/32107630853)
- **分支**: `fix/xpu-compressed-tensors-fp8-w8a8`
- **总耗时**: 45.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32107630853

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 10.5min | 环境问题 | GitHub Actions 下载 actions/checkout 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620284945) |
| base-b-test-16-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285050) |
| base-b-test-1-npu-a3 / run (0) | 2.0min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285069) |
| base-b-test-8-npu-a3 / run (0) | 3.3min | 环境问题 | GitHub Actions 下载 actions/checkout 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285096) |
| base-b-test-4-npu-a3 / run (1) | 2.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285117) |
| base-b-test-2-npu-a3 / run (0) | 1.1min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285146) |
| base-b-test-4-npu-a3 / run (0) | 3.3min | 其他 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285162) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.9min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285243) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查快速失败，因其他作业失败导致本作业被跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285318) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 1.8min | 其他 | PR健康检查失败，因其他根因作业失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285359) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.0min | 其他 | 健康检查快速失败，因其他作业失败被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285361) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/checkout 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功获取，属于网络或 GitHub 服务问题，非代码或测试本身错误。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620284945

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因作业为multimodal-gen-test-1-npu-a3，本作业因快速失败被终止，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285050

- **base-b-test-1-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被过滤并快速失败，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285069

- **base-b-test-8-npu-a3 / run (0)**: 日志显示下载 actions/checkout@v4 时因 HttpClient.Timeout 100秒超时而失败，重试后仍未能成功，最终导致作业无法继续执行。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285096

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，导致本作业未实际运行即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285117

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285146

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，最终根因是multimodal-gen-test-1-npu-a3作业失败，本作业因fast-fail机制被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285162

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285243

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查发现multimodal-gen-test-1-npu-a3作业失败，触发fast-fail机制，本作业未实际运行即被跳过，属于关联失败而非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285318

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285359

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 该作业在“Check PR test health”步骤被快速失败机制跳过，根因是multimodal-gen-test-1-npu-a3作业失败，本作业并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285361

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.4min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32107630853/job/95620285109) |


## [Run #32107345516](https://github.com/sgl-project/sglang/actions/runs/32107345516)
- **分支**: `main`
- **总耗时**: 15.7min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32107345516

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 14.3min | 环境问题 | GitHub Actions 下载 upload-artifact 超时，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107345516/job/95619439328) |

- **multimodal-gen-test-1-npu-a3**: 日志显示下载 actions/upload-artifact@v4 时因 HttpClient.Timeout 100秒超时而失败，虽然后续重试成功，但可能影响作业稳定性。此外 Node 20 弃用警告提示环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107345516/job/95619439328


## [Run #32107054597](https://github.com/sgl-project/sglang/actions/runs/32107054597)
- **分支**: `main`
- **总耗时**: 13.0min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32107054597

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 7.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601542) |
| base-b-test-4-npu-a3 / run (0) | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601614) |
| base-b-test-2-npu-a3 / run (0) | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601671) |
| base-b-test-8-npu-a3 / run (0) | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601689) |
| base-b-test-16-npu-a3 / run (0) | 12.1min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业失败。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601731) |
| base-b-test-1-npu-a3 / run (0) | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601736) |
| base-b-test-4-npu-a3 / run (1) | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601745) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601980) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618602035) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618602073) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 12.1min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取必要资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618602076) |

- **multimodal-gen-test-1-npu-a3**: 日志中只包含GitHub Actions运行器初始化、Node.js弃用警告及上传artifact步骤，未包含multimodal-gen测试的实际执行输出或错误信息，无法判断具体失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601542

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的构建产物或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601614

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查上传步骤或资源生命周期。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601671

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601689

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI系统尝试下载或访问的工件/日志文件在存储中缺失，可能是上传失败、路径错误或过期清理所致，属于基础设施环境问题，与代码或性能无关。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601731

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601736

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601745

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601980

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件缺失或路径错误，可能是资源未上传或已被删除，属于环境配置问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618602035

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置变更所致，需检查相关存储路径。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618602073

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618602076

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.9min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32107054597/job/95618601604) |


## [Run #32106864860](https://github.com/sgl-project/sglang/actions/runs/32106864860)
- **分支**: `experiment/h3-single-gpu-offload`
- **总耗时**: 15.6min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32106864860

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 13.1min | 其他 | 作业日志被截断，未显示实际测试结果，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106864860/job/95618066427) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法看到测试执行的具体输出。从可见内容看，作业正常启动并完成，上传diffusion-failures目录时提示无文件，说明测试可能通过或未产生失败产物，但无法确认最终状态。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106864860/job/95618066427


## [Run #32106843246](https://github.com/sgl-project/sglang/actions/runs/32106843246)
- **分支**: `agentx-upstream/query-prefill-overlap-20260816`
- **总耗时**: 68.1min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32106843246

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 27.8min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和清理信息。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063633) |
| base-b-test-8-npu-a3 / run (0) | 5.2min | 其他 | 健康检查发现根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063733) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063736) |
| base-b-test-4-npu-a3 / run (0) | 2.2min | 其他 | 健康检查快速失败，因其他作业根因失败导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063762) |
| base-b-test-16-npu-a3 / run (0) | 1.7min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063783) |
| base-b-test-2-npu-a3 / run (0) | 0.7min | 环境问题 | 健康检查发现其他作业失败导致本作业被快速跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063871) |
| base-b-test-4-npu-a3 / run (1) | 2.2min | 环境问题 | 健康检查发现根因作业multimodal-gen-test-1-npu-a3失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063910) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 0.8min | 其他 | 该作业因其他根因作业失败而被快速失败跳过，并非自身问题。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625064183) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625064274) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 7.7min | 其他 | 健康检查发现根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625064358) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.7min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625064375) |

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行阶段的输出，仅有runner启动、依赖下载、上传artifact（无文件）及清理步骤，无法判断具体失败原因，可能为日志截断或作业在测试前被中断。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063633

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因作业multimodal-gen-test-1-npu-a3失败，本作业作为级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063733

- **base-b-test-1-npu-a3 / run (0)**: 日志显示health-check检测到multimodal-gen-test-1-npu-a3作业失败，属于根因失败，因此本作业（base-b-test-1-npu-a3）被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063736

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业被标记为级联失败并快速跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063762

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被Fast-fail跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063783

- **base-b-test-2-npu-a3 / run (0)**: 作业在启动阶段执行健康检查时，检测到同一次PR运行中的multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063871

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因fast-fail机制被跳过，并非自身代码或环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063910

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 健康检查发现根因失败作业为multimodal-gen-test-1-npu-a3，本作业被级联跳过，日志中无自身测试执行记录。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625064183

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625064274

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，导致本作业被快速失败跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625064358

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625064375

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.3min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32106843246/job/95625063724) |


## [Run #32106333853](https://github.com/sgl-project/sglang/actions/runs/32106333853)
- **分支**: `unidy2002/weight-cache-static-dp-ep`
- **总耗时**: 59.4min | **结论**: failure
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32106333853

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 23.6min | 其他 | 作业日志被截断，未显示实际测试结果，仅见上传工件时无失败文件。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502607) |
| base-b-test-16-npu-a3 / run (0) | 5.8min | 其他 | 健康检查发现其他作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502740) |
| base-b-test-4-npu-a3 / run (1) | 3.7min | 其他 | 健康检查发现根因任务失败，触发快速失败机制，导致本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502755) |
| base-b-test-4-npu-a3 / run (0) | 2.5min | 环境问题 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502842) |
| base-b-test-2-npu-a3 / run (0) | 2.9min | 环境问题 | 健康检查发现根因作业失败，导致本作业被级联跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502851) |
| base-b-test-1-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502870) |
| base-b-test-8-npu-a3 / run (0) | 0.8min | 其他 | 健康检查发现其他根因作业失败，本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502881) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，触发快速失败机制，本作业被跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616503047) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 0.8min | 其他 | 健康检查发现其他根因作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616503081) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 2.1min | 其他 | 健康检查发现其他作业失败，导致本作业被快速失败跳过。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616503136) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 1.4min | 其他 | PR测试健康检查失败，根因是多模态生成测试失败导致级联跳过 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616503210) |

- **multimodal-gen-test-1-npu-a3**: 日志中间部分被省略，无法判断测试是否通过或失败。仅看到上传diffusion-failures工件时提示无文件，说明可能没有失败用例，但作业最终状态未知，需查看完整日志。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502607

- **base-b-test-16-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业作为级联失败被快速失败跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502740

- **base-b-test-4-npu-a3 / run (1)**: 日志显示健康检查过滤了多个级联失败任务，最终根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败被终止，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502755

- **base-b-test-4-npu-a3 / run (0)**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502842

- **base-b-test-2-npu-a3 / run (0)**: 日志显示健康检查过滤了多个级联失败作业，根因是multimodal-gen-test-1-npu-a3失败，本作业因快速失败机制被跳过，非自身代码问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502851

- **base-b-test-1-npu-a3 / run (0)**: 作业在启动前的健康检查中检测到multimodal-gen-test-1-npu-a3作业失败，触发了fast-fail机制，本作业未实际运行测试即被终止。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502870

- **base-b-test-8-npu-a3 / run (0)**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502881

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，作为根因导致本作业被快速失败跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616503047

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示健康检查检测到根因失败作业multimodal-gen-test-1-npu-a3，本作业因级联失败被跳过，并非自身测试失败。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616503081

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示健康检查检测到multimodal-gen-test-1-npu-a3作业失败，被判定为根因失败，因此本作业被快速失败机制跳过，未实际执行测试。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616503136

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 该作业在启动前被PR测试健康检查拦截，根因是多模态生成测试失败，导致本作业及多个相关作业被级联跳过，并非本作业自身问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616503210

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 6.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32106333853/job/95616502758) |


## [Run #32106097016](https://github.com/sgl-project/sglang/actions/runs/32106097016)
- **分支**: `main`
- **总耗时**: 5.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32106097016

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| multimodal-gen-test-1-npu-a3 | 4.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32106097016/job/95615898676) |

- **multimodal-gen-test-1-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件缺失或路径错误，可能是资源未上传、被删除或配置有误，属于环境或基础设施问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32106097016/job/95615898676


## [Run #32105615176](https://github.com/sgl-project/sglang/actions/runs/32105615176)
- **分支**: `main`
- **总耗时**: 7.2min | **结论**: cancelled
- **workflow 链接**: https://github.com/sgl-project/sglang/actions/runs/32105615176

### ⚠️ 失败/超时任务

| 任务 | 耗时 | 分类 | AI 分析 | 链接 |
|------|------|------|---------|------|
| base-b-test-8-npu-a3 / run (0) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413091) |
| base-b-test-16-npu-a3 / run (0) | 6.3min | 环境问题 | 日志显示Azure Blob存储中指定的blob不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413111) |
| base-b-test-2-npu-a3 / run (0) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413122) |
| base-b-test-4-npu-a3 / run (1) | 6.3min | 环境问题 | Azure Blob 存储中指定的文件不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413127) |
| base-b-test-4-npu-a3 / run (0) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413138) |
| multimodal-gen-test-1-npu-a3 | 6.0min | 其他 | 作业日志不完整，未显示实际测试失败原因，仅包含环境准备和上传工件步骤。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413154) |
| base-b-test-1-npu-a3 / run (0) | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413181) |
| base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413328) |
| base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413351) |
| base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413393) |
| base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3 | 6.3min | 环境问题 | Azure Blob 存储中指定的 blob 不存在，导致作业无法获取所需资源。 | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413436) |

- **base-b-test-8-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 尝试下载的工件或缓存文件在 Azure Blob 中已被删除或路径错误，可能是资源过期或配置问题，需检查存储路径或重新上传。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413091

- **base-b-test-16-npu-a3 / run (0)**: 错误码BlobNotFound表明CI作业尝试下载的工件或缓存文件在存储中缺失，可能是文件被清理、路径错误或上传失败，属于基础设施环境问题。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413111

- **base-b-test-2-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413122

- **base-b-test-4-npu-a3 / run (1)**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个 blob 文件已被删除或路径错误，可能是缓存或上传步骤失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413127

- **base-b-test-4-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的构建产物或缓存文件在 Azure Blob 存储中已被删除或路径错误，属于环境或资源缺失问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413138

- **multimodal-gen-test-1-npu-a3**: 日志中未包含测试执行或失败的具体信息，只有GitHub Actions环境准备、Node版本警告及上传diffusion-failures工件时未找到文件的提示，无法判断测试失败原因。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413154

- **base-b-test-1-npu-a3 / run (0)**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载或访问的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传文件。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413181

- **base-c-test-acc-4-npu-a3 / base-c-test-acc-4-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 作业尝试下载的 Azure Blob 文件已被删除或路径错误，可能是资源清理或配置问题，需检查存储路径或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413328

- **base-c-test-acc-2-npu-a3 / base-c-test-acc-2-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源未上传或已被删除，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413351

- **base-c-test-acc-8-npu-a3 / base-c-test-acc-8-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的某个文件或工件在 Azure Blob 存储中缺失或路径错误，可能是资源被清理或上传失败，需检查相关存储配置。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413393

- **base-c-test-acc-16-npu-a3 / base-c-test-acc-16-npu-a3**: 日志显示 BlobNotFound 错误，说明 CI 依赖的远程文件（如模型权重或缓存）已被删除或路径错误，需检查存储配置或重新上传资源。
  链接: https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413436

### 正常任务

| 任务 | 耗时 | 结果 | 链接 |
|------|------|------|------|
| base-a-test-1-npu-a2 / run (0) | 5.2min | success | [job link](https://github.com/sgl-project/sglang/actions/runs/32105615176/job/95614413084) |


---
*Auto-generated by npu_pr_monitor.py*